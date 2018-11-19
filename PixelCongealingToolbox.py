from EntropyEstimator import EntropyEstimator
from ExplodedEntropyEstimator import ExplodedEntropyEstimator
import time, logging
from timer import Timer
from skimage import io, color
import numpy as np
from scipy.ndimage.filters import gaussian_filter

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
        self.rot_step = (1.0/f_pixels)/(1.0+(W/(2.0*f_pixels)**2))/2.0
        self.trans_step = 1.0/(2.0*f_pixels)
        self.utils = utils
        self.estimator = ExplodedEntropyEstimator()

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
    
    def tweak_ABC_parameters(self, src_images, cur_images, params, dtheta, n_bins, sequential=False, percentage=None):
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
        dtheta = dtheta*10

        A_transforms_minus = np.copy(params).reshape([self.T,6])
        A_transforms_minus[:,0] -= dtheta
        A_images_minus,_ = self.utils.update_stacks(np.repeat(A_transforms_minus, n_bins, axis=0), src_images, 9)

        B_transforms_minus = np.copy(params).reshape([self.T,6])
        B_transforms_minus[:,1] -= dtheta
        B_images_minus,_ = self.utils.update_stacks(np.repeat(B_transforms_minus, n_bins, axis=0), src_images, 9)

        C_transforms_minus = np.copy(params).reshape([self.T,6])
        C_transforms_minus[:,2] -= dtheta
        C_images_minus,_ = self.utils.update_stacks(np.repeat(C_transforms_minus, n_bins, axis=0), src_images, 9)

        A_transforms_plus = np.copy(params).reshape([self.T,6])
        A_transforms_plus[:,0] += dtheta
        A_images_plus,_ = self.utils.update_stacks(np.repeat(A_transforms_plus, n_bins, axis=0), src_images, 9)

        B_transforms_plus = np.copy(params).reshape([self.T,6])
        B_transforms_plus[:,1] += dtheta
        B_images_plus,_ = self.utils.update_stacks(np.repeat(B_transforms_plus, n_bins, axis=0), src_images, 9)

        C_transforms_plus = np.copy(params).reshape([self.T,6])
        C_transforms_plus[:,2] += dtheta
        C_images_plus,_ = self.utils.update_stacks(np.repeat(C_transforms_plus, n_bins, axis=0), src_images, 9)

        for i in range(self.T):
            current_image = cur_images[i*n_bins:(i+1)*n_bins,...]

            new_image1 = A_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent1, _ = self.estimator.get_updated_entropy(current_image, new_image1, percentage)

            new_image2 = B_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent2, _ = self.estimator.get_updated_entropy(current_image, new_image2, percentage)

            new_image3 = C_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent3, _ = self.estimator.get_updated_entropy(current_image, new_image3, percentage)

            ent0, _ = self.estimator.get_updated_entropy(current_image, current_image, percentage)

            new_image4 = A_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent4, _ = self.estimator.get_updated_entropy(current_image, new_image4, percentage)

            new_image5 = B_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent5, _ = self.estimator.get_updated_entropy(current_image, new_image5, percentage)

            new_image6 = C_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent6, _ = self.estimator.get_updated_entropy(current_image, new_image6, percentage)

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

    def tweak_UVW_parameters(self, src_images, cur_images, params, dtrans, n_bins, sequential=False):
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
        dtrans = dtrans*5

        U_transforms_minus = np.copy(params).reshape([self.T,6])
        U_transforms_minus[:,3] -= dtrans
        U_images_minus,_ = self.utils.update_stacks(np.repeat(U_transforms_minus, n_bins, axis=0), src_images, 9)

        V_transforms_minus = np.copy(params).reshape([self.T,6])
        V_transforms_minus[:,4] -= dtrans
        V_images_minus,_ = self.utils.update_stacks(np.repeat(V_transforms_minus, n_bins, axis=0), src_images, 9)

        W_transforms_minus = np.copy(params).reshape([self.T,6])
        W_transforms_minus[:,5] -= dtrans
        W_images_minus,_ = self.utils.update_stacks(np.repeat(W_transforms_minus, n_bins, axis=0), src_images, 9)

        U_transforms_plus = np.copy(params).reshape([self.T,6])
        U_transforms_plus[:,3] += dtrans
        U_images_plus,_ = self.utils.update_stacks(np.repeat(U_transforms_plus, n_bins, axis=0), src_images, 9)

        V_transforms_plus = np.copy(params).reshape([self.T,6])
        V_transforms_plus[:,4] += dtrans
        V_images_plus,_ = self.utils.update_stacks(np.repeat(V_transforms_plus, n_bins, axis=0), src_images, 9)

        W_transforms_plus = np.copy(params).reshape([self.T,6])
        W_transforms_plus[:,5] += dtrans
        W_images_plus,_ = self.utils.update_stacks(np.repeat(W_transforms_plus, n_bins, axis=0), src_images, 9)

        for i in range(self.T):
            current_image = cur_images[i*n_bins:(i+1)*n_bins,...]

            new_image1 = U_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent1, _ = self.estimator.get_updated_entropy(current_image, new_image1)

            new_image2 = V_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent2, _ = self.estimator.get_updated_entropy(current_image, new_image2)

            new_image3 = W_images_minus[i*n_bins:(i+1)*n_bins,...]
            ent3, _ = self.estimator.get_updated_entropy(current_image, new_image3)

            new_image0 = current_image
            ent0, _ = self.estimator.get_updated_entropy(current_image, new_image0)

            new_image4 = U_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent4, _ = self.estimator.get_updated_entropy(current_image, new_image4)

            new_image5 = V_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent5, _ = self.estimator.get_updated_entropy(current_image, new_image5)

            new_image6 = W_images_plus[i*n_bins:(i+1)*n_bins,...]
            ent6, _ = self.estimator.get_updated_entropy(current_image, new_image6)

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

    def tweak_inv_depth(self, src_images, cur_images, params, inv_dep_map, n_bins, level, sample, sequential=False):

        H = cur_images.shape[1]
        W = cur_images.shape[2]

        dZ = 0.04
        num = 3
        choices = np.zeros((num,H*W))
        changes = np.zeros((num,H,W))
        prod = np.linspace(-0.2, 0.2, num)
        for i in range(num):
            inv_dep_map_temp = (1.0+prod[i]) * inv_dep_map
            changes[i] = inv_dep_map_temp
            self.utils.update_inv_dep_map(inv_dep_map_temp)
            self.utils.pick_sample(sample)
            cur_images,_ = self.utils.update_stacks(np.repeat(params, n_bins, axis=0), src_images, 9)
            self.estimator.build_histogram(cur_images, (1, 256), n_bins)
            entropy_temp, lin_entropy_temp = self.estimator.compute_entropy()
            choices[i] = lin_entropy_temp

        # print 'choices'
        # print choices.reshape((num,H,W))[:,H/2-5:H/2+5,W/2-5:W/2+5]

        # inv_dep_map1 = (1.0-0.1) * inv_dep_map
        # self.utils.update_inv_dep_map(inv_dep_map1)
        # self.utils.pick_sample(sample)
        # cur_images, _ = self.utils.update_stacks(params, src_images, 9)
        # self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        # entropy1, lin_entropy1 = self.estimator.compute_entropy()

        # self.utils.update_inv_dep_map(inv_dep_map)
        # self.utils.pick_sample(sample)
        # cur_images, _ = self.utils.update_stacks(params, src_images, 9)
        # self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        # entropy0, lin_entropy0 = self.estimator.compute_entropy()

        # inv_dep_map2 = (1.0+0.1) * inv_dep_map
        # self.utils.update_inv_dep_map(inv_dep_map2)
        # self.utils.pick_sample(sample)
        # cur_images, _ = self.utils.update_stacks(params, src_images, 9)
        # self.estimator.build_histogram(cur_images, (1, 256), n_bins)
        # entropy2, lin_entropy2 = self.estimator.compute_entropy()

        #choices = np.array([lin_entropy1, lin_entropy0, lin_entropy2])

        index = np.argmin(choices, axis=0).reshape((H,W))

        # print 'index'
        # print index[H/2-5:H/2+5,W/2-5:W/2+5]

        #changes = np.array([inv_dep_map1, inv_dep_map, inv_dep_map2])
        sharp_map = np.copy(inv_dep_map)
        for i in range(H):
            for j in range(W):
                sharp_map[i,j] = changes[index[i,j],i,j]
 
        blurred_map = np.ones((H/36,W/48))
        for i in range(H/36):
            for j in range(W/48):
                blurred_map[i,j] = np.mean(sharp_map[i*36:(i+1)*36,j*48:(j+1)*48])
        blurred_map = np.repeat(np.repeat(blurred_map,48,axis=1),36,axis=0)
#        blurred_map = gaussian_filter(sharp_map, sigma=20, truncate=(((51 - 1)/2)-0.5)/50)

        return blurred_map

    def create_depth_image(self, inv_dep_map):
        temp = np.copy(inv_dep_map)
        temp = (temp-np.median(temp))
        im = 255*(1. / (1 + np.exp(-temp+1)))
        return im

    def exploding(self, gray_stacks, num_bins, s, w):
        T, H, W, _ = gray_stacks.shape
        exploded_data = np.zeros((T, num_bins, H, W, 1))
        filtered_data = np.zeros((T, num_bins, H, W, 1))
        judge = np.linspace(1.0, 256.0, num=num_bins+1)
        for i in range(num_bins):
            exploded_data[:,i,...] = (gray_stacks >= judge[i])*(gray_stacks < judge[i+1])
            t = (((w - 1)/2)-0.5)/s
            filtered_data[:,i,...] = gaussian_filter(exploded_data[:,i,...], sigma=s, truncate=t)*255

        return filtered_data

    def coordinate_descent(self, parameters, gray_stacks, config, level, sample=1, output_directory=None):
        assert config['PIXEL CONGEAL']['OPTIM']=='coord', "Wrong function call for optimizer"

        n_bins = int(config['PIXEL CONGEAL']['NBINS'])
        n_iters = int(config['PIXEL CONGEAL']['ITERS'])
        percentage = float(config['PIXEL CONGEAL']['PERCENTAGE'])
        log.info("Params for pixel congealing: N_BINS: %s, N_ITERS: %s, level: %s, pct: %s" %(n_bins, n_iters, level, percentage))

        #current_stacks = np.copy(gray_stacks)
        expl_timer = Timer()
        expl_timer.tic()
        exploded = self.exploding(gray_stacks, n_bins, 2, 11)
        print 'exploding time ' + str(expl_timer.toc()) + ' seconds'
        for i in range(n_bins):
            io.imsave(output_directory + '/overfiltered_' + str(i) + '.png', color.gray2rgb(exploded[0,i,:,:,0]).astype(np.uint8))
        
        exploded_gray_stacks = exploded.reshape((self.T*n_bins,self.H/sample,self.W/sample,1))
        current_stacks = np.zeros((self.T*n_bins,self.panH/sample,self.panW/sample,1))
        #params = np.copy(parameters)
        params = np.load('../Output/DJIOutput/DJI25/PC_Params/params99.npy')
        inv_dep_map = self.utils.pick_sample(sample)
        start = time.time()
        current_stacks, _ = self.utils.update_stacks(np.repeat(params, n_bins, axis=0), exploded_gray_stacks, 9)
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
            log.info('\n')
            log.info("Pixel congealing iteration " + str(k).zfill(2) + " started")
            it_timer = Timer()
            it_timer.tic()
            dtheta = self.rot_step * level
            dtrans = 2 * self.trans_step * level
            #dZ = 1.0/(init_level + 1) * level
            dZ = 0.1
            log.info("Params for updating step: dtheta: %s, dtrans: %s, dZ: %s" %(dtheta, dtrans, dZ))
            message = ''

            prev_params = np.copy(params)
            prev_entropy = np.copy(entropy)
            prev_inv_dep_map = np.copy(inv_dep_map)

            params = self.tweak_ABC_parameters(exploded_gray_stacks, current_stacks, params, dtheta*4, n_bins)
            params = self.tweak_UVW_parameters(exploded_gray_stacks, current_stacks, params, dtrans*4, n_bins)
            if k==11:
                exploded = self.exploding(gray_stacks, n_bins, 10, 41)
            if k>10:
                inv_dep_map = self.tweak_inv_depth(exploded_gray_stacks, current_stacks, params, inv_dep_map, n_bins, level, sample)
            params = params-np.mean(params, axis=0)

            self.utils.update_inv_dep_map(inv_dep_map)
            self.utils.pick_sample(sample)
            current_stacks, _ = self.utils.update_stacks(np.repeat(params, n_bins, axis=0), exploded_gray_stacks, 9)

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

            it_sapn = it_timer.toc()

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

            print str(k) + 'iterations in ' + str(it_sapn) + ' seconds' 

            self.utils.update_inv_dep_map(inv_dep_map)
            self.utils.pick_sample(sample)
            img_stacks, _ = self.utils.update_stacks(params, gray_stacks, 9)

            if output_directory is not None:
                average_image = np.mean(img_stacks, axis=0)
                io.imsave(output_directory + '/depth.png', color.gray2rgb(self.create_depth_image(inv_dep_map)).astype(np.uint8))
                np.save(output_directory + '/depth.npy', inv_dep_map)
                io.imsave(output_directory + '/PC_process/PC_' + str(k) + '.png', color.gray2rgb(average_image[:,:,0]).astype(np.uint8))
                np.save(output_directory + '/PC_Params/params' + str(k) + '.npy', params)

        inv_dep_map = self.utils.update_inv_dep_map(inv_dep_map)

        return params, inv_dep_map, total_entropy_history
