from EntropyEstimator import EntropyEstimator
import time, logging
from timer import Timer
from skimage import io, color
import numpy as np

log = logging.getLogger(__name__)


class PCToolbox:
    """The tool box for pixel congealing."""

    def __init__(self, H, W, C, T, f_pixels, panH, panW, utils=None):
        """
        Constructor
        :param H: height of the images
        :param W: width of the images
        :param C: number of channels
        :param T: number of frames
        :param K: number of tracks
        :param f_pixels: focal length in pixels
        """
        self.H = H
        self.W = W
        self.shape = H * W
        self.C = C
        self.T = T
        self.panH = panH
        self.panW = panW

        self.f_pixels = f_pixels
        self.rot_step = (1/f_pixels)/(1+(W/(2*f_pixels)**2))/2
        self.trans_step = 1.0/(2.0*f_pixels)
        self.utils = utils
        self.estimator = EntropyEstimator()

    def radical(self, gray_stacks):
        """
        compute z-space entropy
        :param gray_stacks: gray pixel stacks - numpy array (T, H, W, 1) 
        :return zSpace: z space entropy of each pixel
        :return mean_zSpace: the mean and normalized value for z-space
        """
        sort_gray_stacks = np.sort(gray_stacks[:,:,:,0],axis=0)
        N = self.T
        m = np.rint(np.sqrt(N)).astype(np.uint8)
        zSpace = np.log((N+1)/m * (1.0 + sort_gray_stacks[m:N,:,:] - sort_gray_stacks[0:N-m,:,:]))
        zSpace = np.mean(zSpace,axis=0)
        mean_zSpace = 1.0/(N-m) * np.sum(zSpace) / (self.shape)
        return zSpace, mean_zSpace
        
    def get_zSpace_mask(self, zSpace, percentage=0.5):
        """
        get z-space mask
        :param zSpace: computed z space values - numpy array (H, W) 
        :percentage: How many well aligned pixels to use
        :return: linear pixels mask of z-space - A numpy boolean array of shape (H*W,)
        """
        assert 0<percentage<=1, 'Invalid percentage value for mask'

        lin_zSpace = zSpace.flatten()

        sort_lin_zSpace = np.sort(lin_zSpace)
        
        threshold = int(self.shape * percentage)

        lin_mask = lin_zSpace <= sort_lin_zSpace[threshold-1]

        return lin_mask
    
    def get_entropy_mask(self, histogram, percentage=0.5):
        """
        get entropy mask
        :param histogram: computed histogram values - numpy array (H*W, num_bins) 
        :percentage: How many well aligned pixels to use
        :return: linear pixels mask of entropy - A numpy boolean array of shape (H*W,)
        """
        assert 0<percentage<=1, 'Invalid percentage value for mask'

        entropy = np.sum(histogram * np.log(histogram),axis=1)

        lin_entropy = entropy.flatten()
        sort_lin_entropy = np.sort(lin_entropy)
        
        threshold = int(self.shape * percentage)

        lin_mask = lin_entropy >= sort_lin_entropy[threshold-1]

        return lin_mask
    
    def get_nonzero_mask(self, gray_stacks):
        """
        :param gray_stacks: gray pixel stacks - numpy array (T, H, W, 1) 
        :return: linear nonzero pixels mask - A numpy boolean array of shape (H*W,)
        """
        nonzero_mask = np.invert(np.any(gray_stacks == 0, axis=0))
        lin_nonzero_mask = nonzero_mask.flatten()

        return lin_nonzero_mask
    
    def tweak_ABC_parameters(self, src_images, cur_images, params, dtheta, sequential=False, percentage=None):
        """
        tweak the values of rotation parameters based on minimize entropy
        :param src_images: source images
        :param cur_images: current images
        :param params: parameters need to tweak
        :param ori_ent: original entropy before tweak
        :param dtheta: tweak step
        :param sequential: sequential rotation or not
        :param percentage: percentage of values to use from entropy calculation.
        :return: updated params
        """
        changes = np.zeros([self.T, 3])

        A_transforms_minus = np.copy(params).reshape([self.T,6])
        A_transforms_minus[:,0] -= dtheta
        A_images_minus = self.utils.update_stacks(A_transforms_minus, src_images, 9)

        B_transforms_minus = np.copy(params).reshape([self.T,6])
        B_transforms_minus[:,1] -= dtheta
        B_images_minus = self.utils.update_stacks(B_transforms_minus, src_images, 9)

        C_transforms_minus = np.copy(params).reshape([self.T,6])
        C_transforms_minus[:,2] -= dtheta
        C_images_minus = self.utils.update_stacks(C_transforms_minus, src_images, 9)

        A_transforms_plus = np.copy(params).reshape([self.T,6])
        A_transforms_plus[:,0] += dtheta
        A_images_plus = self.utils.update_stacks(A_transforms_plus, src_images, 9)

        B_transforms_plus = np.copy(params).reshape([self.T,6])
        B_transforms_plus[:,1] += dtheta
        B_images_plus = self.utils.update_stacks(B_transforms_plus, src_images, 9)

        C_transforms_plus = np.copy(params).reshape([self.T,6])
        C_transforms_plus[:,2] += dtheta
        C_images_plus = self.utils.update_stacks(C_transforms_plus, src_images, 9)

        for i in range(self.T):
            current_image = cur_images[i,...]

            new_image1 = A_transforms_minus[i,...]
            ent1, _ = self.estimator.get_updated_entropy(current_image, new_image1, percentage)

            new_image2 = B_transforms_minus[i,...]
            ent2, _ = self.estimator.get_updated_entropy(current_image, new_image2, percentage)

            new_image2 = C_transforms_minus[i,...]
            ent3, _ = self.estimator.get_updated_entropy(current_image, new_image3, percentage)

            ent0, _ = self.estimator.get_updated_entropy(current_image, current_image, percentage)

            new_image4 = A_transforms_plus[i,...]
            ent4, _ = self.estimator.get_updated_entropy(current_image, new_image4, percentage)

            new_image5 = B_transforms_plus[i,...]
            ent5, _ = self.estimator.get_updated_entropy(current_image, new_image5, percentage)

            new_image6 = C_transforms_plus[i,...]
            ent6, _ = self.estimator.get_updated_entropy(current_image, new_image6, percentage)

            this_transform[:,2] -= dtheta

            choices = [ent3, ent2, ent1, ent0, ent4, ent5, ent6]

            delta = choices.index(min(choices))-3
            sign = np.sign(delta)
            axis = np.absolute(delta)
            if axis == 0:
                changes[i] = np.array([0,0,0])*dtheta
            else:
                changes[i, axis-1] = sign*dtheta

        params[:,0:3] = params[:,0:3] + changes

        return params

    def tweak_UVW_parameters(self, src_images, cur_images, params, dtrans, sequential=False):
        """
        tweak the values of rotation parameters based on minimize entropy
        :param src_images: source images
        :param cur_images: current images
        :param params: parameters need to tweak
        :param ori_ent: original entropy before tweak
        :param dtheta: tweak step
        :param sequential: sequential rotation or not
        :param percentage: percentage of values to use from entropy calculation.
        :return: updated params
        """
        changes = np.zeros([self.T, 3])

        U_transforms_minus = np.copy(params).reshape([self.T,6])
        U_transforms_minus[:,0] -= dtrans
        U_images_minus = self.utils.update_stacks(U_transforms_minus, src_images, 9)

        V_transforms_minus = np.copy(params).reshape([self.T,6])
        V_transforms_minus[:,1] -= dtrans
        V_images_minus = self.utils.update_stacks(V_transforms_minus, src_images, 9)

        W_transforms_minus = np.copy(params).reshape([self.T,6])
        W_transforms_minus[:,2] -= dtrans
        W_images_minus = self.utils.update_stacks(W_transforms_minus, src_images, 9)

        U_transforms_plus = np.copy(params).reshape([self.T,6])
        U_transforms_plus[:,0] += dtrans
        U_images_plus = self.utils.update_stacks(U_transforms_plus, src_images, 9)

        V_transforms_plus = np.copy(params).reshape([self.T,6])
        V_transforms_plus[:,1] += dtrans
        V_images_plus = self.utils.update_stacks(V_transforms_plus, src_images, 9)

        W_transforms_plus = np.copy(params).reshape([self.T,6])
        W_transforms_plus[:,2] += dtrans
        W_images_plus = self.utils.update_stacks(W_transforms_plus, src_images, 9)

        for i in range(self.T):
            current_image = cur_images[i,...]

            new_image1 = U_transforms_minus[i,...]
            ent1, _ = self.estimator.get_updated_entropy(current_image, new_image1, 1)

            new_image2 = V_transforms_minus[i,...]
            ent2, _ = self.estimator.get_updated_entropy(current_image, new_image2, 1)

            new_image3 = W_transforms_minus[i,...]
            ent3, _ = self.estimator.get_updated_entropy(current_image, new_image3, 1)

            new_image0 = current_image
            ent0, _ = self.estimator.get_updated_entropy(current_image, new_image0, 1)

            new_image4 = U_transforms_plus[i,...]
            ent4, _ = self.estimator.get_updated_entropy(current_image, new_image4, 1)

            new_image5 = V_transforms_plus[i,...]
            ent5, _ = self.estimator.get_updated_entropy(current_image, new_image5, 1)

            new_image6 = W_transforms_plus[i,...]
            ent6, _ = self.estimator.get_updated_entropy(current_image, new_image6, 1)

            choices = [ent3, ent2, ent1, ent0, ent4, ent5, ent6]

            delta = choices.index(min(choices))-3
            sign = np.sign(delta)
            axis = np.absolute(delta)
            if axis == 0:
                changes[i] = np.array([0,0,0])*dtrans
            else:
                changes[i, axis-1] = sign*dtrans

        params[:,3:6] = params[:,3:6] + changes

        return params

    def tweak_inv_depth(self, src_images, cur_images, params, inv_dep_map, n_bins, dZ, sample, sequential=False):

        H = cur_images.shape[1]
        W = cur_images.shape[2]

        inv_dep_map1 = (1.0-dZ) * inv_dep_map
        self.utils.update_inv_dep_map(inv_dep_map1)
        self.utils.pick_sample(sample)
        cur_images = self.utils.update_stacks(params, src_images, 9)
        self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        entropy1, lin_entropy1 = self.estimator.compute_entropy()

        self.utils.update_inv_dep_map(inv_dep_map)
        self.utils.pick_sample(sample)
        cur_images = self.utils.update_stacks(params, src_images, 9)
        self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        entropy0, lin_entropy0 = self.estimator.compute_entropy()

        inv_dep_map2 = (1.0+dZ) * inv_dep_map
        self.utils.update_inv_dep_map(inv_dep_map2)
        self.utils.pick_sample(sample)
        cur_images = self.utils.update_stacks(params, src_images, 9)
        self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        entropy2, lin_entropy2 = self.estimator.compute_entropy()

        choices = np.array([lin_entropy1, lin_entropy0, lin_entropy2])

        index = np.argmin(choices, axis=0).reshape((H,W))

        changes = np.array([inv_dep_map1, inv_dep_map, inv_dep_map2])
        for i in range(H):
            for j in range(W):
                inv_dep_map[i,j] = changes[index[i,j],i,j]

        return inv_dep_map

    def coordinate_descent(self, parameters, gray_stacks, config, level, sample=1, output_directory=None):
        assert config['PIXEL CONGEAL']['OPTIM']=='coord', "Wrong function call for optimizer"

        n_bins = int(config['PIXEL CONGEAL']['NBINS'])
        n_iters = int(config['PIXEL CONGEAL']['ITERS'])
        percentage = float(config['PIXEL CONGEAL']['PERCENTAGE'])
        log.info("Params for pixel congealing: N_BINS: %s, N_ITERS: %s, level: %s, pct: %s" %(n_bins, n_iters, level, percentage))

        #current_stacks = np.copy(gray_stacks)
        current_stacks = np.zeros((self.T,self.panH/sample,self.panW/sample,1))
        params = np.copy(parameters)
        inv_dep_map = self.utils.pick_sample(sample)
        start = time.time()
        current_stacks = self.utils.update_stacks(params, gray_stacks, 9)
        stop = time.time()
        duration = str(stop-start)
        log.info('%s seconds to update all images'%duration)

        total_entropy_history = []
        start = time.time()
        self.estimator.build_histogram(current_stacks, (1, 256), n_bins)
        stop= time.time()
        duration = str(stop-start)
        log.info('%s seconds to compute histogram'%duration)
        entropy, lin_entropy = self.estimator.compute_entropy()

        level = 2**level
        init_level = level
	
        for k in range(n_iters):
            log.info("\n Pixel congealing iteration " + str(k) + "started")
            it_timer = Timer()
            it_timer.tic()
            dtheta = self.rot_step * level
            dtrans = 2 * self.trans_step * level
            dZ = 1.0/(init_level + 1) * level
            log.info("Params for updating step: dtheta: %s, dtrans: %s, dZ: %s" %(dtheta, dtrans, dZ))
            log.info(str(k).zfill(2))
            message = ''

            prev_params = np.copy(params)
            prev_entropy = np.copy(entropy)
            prev_inv_dep_map = np.copy(inv_dep_map)

            params = self.tweak_ABC_parameters(gray_stacks, current_stacks, params, 
                      dtheta, percentage=1)
            params = self.tweak_UVW_parameters(gray_stacks, current_stacks, params, 
                      dtrans)
            inv_dep_map = self.tweak_inv_depth(gray_stacks, current_stacks, params, 
                      inv_dep_map, n_bins, dZ, sample)
            params = params-np.mean(params, axis=0)

            self.utils.update_inv_dep_map(inv_dep_map)
            self.utils.pick_sample(sample)
            current_stacks = self.utils.update_stacks(params, gray_stacks, iteration)
            #current_stacks = self.utils.update_stacks(gray_stacks, params, current_stacks)

            self.estimator.build_histogram(current_stacks, (1, 256), n_bins)

            entropy, lin_entropy = self.estimator.compute_entropy()
            improvement = prev_entropy - entropy

            if (k%10==0 and k>0) or (k>3 and len(np.unique(total_entropy_history[-3:]))==1) or (improvement < 1e-7):
                level = level-1
                params = prev_params
                inv_dep_map = prev_inv_dep_map
                message += '\nReducing step size to:' + str(dtheta) + str(dtrans) + str(dZ)

            total_entropy_history.append(entropy)
            message += '\n' + 'Entropy: ' + str(entropy) + '\n' + 'Level: ' + str(level)

            it_sapn = it_timer.tic()

            message += '\n'+ str(it_sapn) + ' seconds'
            if level < 0 :
                break
                # if init_level > 0 :
                #     #break
                #     init_level = int(init_level/2.0)
                #     level = init_level
                # else:
                #     break
            log.info(message)
            log.info('A: ' + str(np.sum(np.absolute(params),axis=0)[0]) + \
                    ' B: ' + str(np.sum(np.absolute(params),axis=0)[1]) + \
                    ' C: ' + str(np.sum(np.absolute(params),axis=0)[2]) + \
                    ' U: ' + str(np.sum(np.absolute(params),axis=0)[3]) + \
                    ' V: ' + str(np.sum(np.absolute(params),axis=0)[4]) + \
                    ' W: ' + str(np.sum(np.absolute(params),axis=0)[5]))

            if output_directory is not None:
                average_image = np.mean(current_stacks, axis=0)
                io.imsave(output_directory + '/PC_process/PC_' + str(k) + '.png', color.gray2rgb(average_image[:,:,0]).astype(np.uint8))
                np.savetxt(output_directory + '/PC_Params/params' + str(k) + '.npy', params, delimiter=",")

        inv_dep_map = self.utils.update_inv_dep_map(inv_dep_map)

        return params, inv_dep_map, total_entropy_history
