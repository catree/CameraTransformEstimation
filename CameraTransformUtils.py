import numpy as np
from HornToolbox import HornToolbox
from skimage.measure import block_reduce

class CRUtils:

    def __init__(self, H=0, W=0, f_pixels=0, panH=0, panW=0, toolbox='Horn'):
        """
        Constructor

        params: H: the input image size H.
        params: W: the input image size W.
        params: C: the input image color channels.
        params: f_pixels: the input image focal length of pixels.
        params: panH: the panorama image size H.
        params: panW: the panorama image size W.
        params: toolbox: the type of toolbox to compute transformations. selected from {'Horn','Euler','Quart'}
        """
        self.H = H
        self.W = W
        self.f_pixels = f_pixels
        self.panH = panH
        self.panW = panW
        self.inv_dep_map = np.ones((self.panH,self.panW))
        self.sample = 1
        self.sH = self.panH/self.sample
        self.sW = self.panW/self.sample
        
        self.pick_sample(self.sample)
        if(toolbox=='Horn')
            self.toolbox = HornToolbox()
        else if(toolbox=='Euler')
            self.toolbox = EulerToolbox()
        else if(toolbox=='Quart')
            self.toolbox = QuartToolbox()

    def pick_sample(self, sample=1):
        """
        downsample the panorama coordinates.

        params: sample: the sample step to be used, default 1.

        return: the sampled inverse depth map of the panorama.
        """
        self.sample = sample
        self.sH = self.panH/self.sample
        self.sW = self.panW/self.sample
        self.seed_points = np.ones([1, self.sH * self.sW, 3])
        start = int((sample-1)/2)
        x = np.arange(start, self.panW, self.sample)
        y = np.arange(start, self.panH, self.sample)
        xx, yy = np.meshgrid(x, y)
        self.seed_points[:,:,0:2] = np.stack((xx, yy), axis=-1).reshape((1,-1,2))
        self.seed_points[:,:,2] = self.inv_dep_map[self.seed_points[:,:,1].astype(int),self.seed_points[:,:,0].astype(int)]
        self.seed_points_norm = self.normalize_input(self.seed_points)

        return self.seed_points_norm[0, :, 2].reshape((self.sH,self.sW))

    def update_inv_dep_map(self, inv_dep_map):
        """
        update and upsample the inverse depth map using an input inverse depth map

        params: inv_dep_map: the input inverse depth map, [sH, sW]

        return: upsampled inverse depth map, [panH, panW]
        """
        assert inv_dep_map.shape = self.seed_points[0, :, 2].shape, 'wrong input size of inverse depth map in update_inv_dep_map'
        self.seed_points[0, :, 2] = inv_dep_map.reshape((-1))
        self.inv_dep_map = inv_dep_map.repeat(self.sample, axis = 0).repeat(self.sample, axis = 1)

        return self.inv_dep_map


    def normalize_input(self, seed_points):
        """
        normalize the pixel locations using focal length

        :param seed_points: pixel location, [T, K, 3]

        :return: normalized version of the seed_points
        """
        seed_points_norm = np.copy(seed_points).astype(np.float64)

        seed_points_norm[:,:,0] -= (self.panW/2.0 + 0.5)
        seed_points_norm[:,:,1] -= (self.panH/2.0 + 0.5)
        seed_points_norm[:,:,0:2] = seed_points_norm[:,:,0:2]/self.f_pixels

        return seed_points_norm

    def denormalize_input(self, seed_points_norm):
        """
        denormalize the pixel location using focal length

        :param seed_points_norm: pixel location [T, K, 3]

        :return: denormalized version of the seed_points_norm
        """
        seed_points = np.copy(seed_points_norm)

        seed_points[:,:,0:2] = seed_points[:,:,0:2]*self.f_pixels
        seed_points[:,:,0] += (self.panW/2.0) + 0.5
        seed_points[:,:,1] += (self.panH/2.0) + 0.5

        return seed_points

    def backward_transform(self, src, dst, src_points, dst_points, interpolation=False):
        """
        transformation of images by backward copying pixels

        :param src: the source images load from file, [T, H, W ,C]
        :param dst: the destination images formed by backward_transform, [T, sH, sW ,C]
        :param src_points: the location of pixels in source image, [T, sH*sW, 3], 0 = x co-ord, 1 = y co-ord
        :param dst_points: the location of pixels in destination image, [T, sH*sW, 3], 0 = x co-ord, 1 = y co-ord

        :returns: destination images after applying transformations using backward transform, [T, sH, sW, C]
        :returns: masks after get applying transformations, [T, sH, sW]
        """
        #assert len(dst.shape)==3 and len(src.shape)==3, 'Image dimensions do not match'
        #assert len(dst_points.shape)==3 and len(src_points.shape)==3, 'Dimensions are incorrect'

        T = src.shape[0]
        assert dst.shape[0] == T, 'Invalid dst shape in backward_transform'
        assert src_points.shape[0] == T, 'Invalid src_points shape in backward_transform'
        assert dst_points.shape[0] == T, 'Invalid dst_points shape in backward_transform'
        assert src_points.shape == dst_points.shape, 'src_points shape and dst_points shape does not match'
        K = src_points.shape[1]

        sW = self.sW
        sH = self.sH
        W = self.W
        H = self.H
        panW = self.panW
        panH = self.panH

        layers = np.arange(T)
        layers = np.repeat(layers, K, axis = -1).reshape((T,-1))
                  
        dst_x = dst_points[:,:,0]
        dst_y = dst_points[:,:,1]

        src_x = src_points[:,:,0]-int((panW-W)/2)
        src_y = src_points[:,:,1]-int((panH-H)/2)
        
        mask1 = src_x>=0 
        mask2 = src_x<W-1
        mask3 = src_y>=0
        mask4 = src_y<H-1
        
        validity_mask = mask1*mask2*mask3*mask4
        
        src_x_valid = (src_x[validity_mask]/self.sample).astype(np.int64).flatten()
        src_y_valid = (src_y[validity_mask]/self.sample).astype(np.int64).flatten()
        dst_x_valid = (dst_x[validity_mask]/self.sample).astype(np.int64).flatten()
        dst_y_valid = (dst_y[validity_mask]/self.sample).astype(np.int64).flatten()
        layers_valid = layers[validity_mask].astype(np.int64).flatten()

        if interpolation is True:
            x = src_x_valid
            y = src_y_valid
            x1 = np.floor(x).astype(np.int64)
            x2 = np.ceil(x).astype(np.int64)
            y1 = np.floor(y).astype(np.int64)
            y2 = np.ceil(y).astype(np.int64)

            dst[layers_valid, dst_y_valid, dst_x_valid, :] = \
                                               src[layers_valid,y1,x1,:]*np.transpose([(x2-x)*(y2-y)])+\
                                               src[layers_valid,y1,x2,:]*np.transpose([(x-x1)*(y2-y)])+\
                                               src[layers_valid,y2,x1,:]*np.transpose([(x2-x)*(y-y1)])+\
                                               src[layers_valid,y2,x2,:]*np.transpose([(x-x1)*(y-y1)])

        else:
            #src_x_valid = np.rint(src_x_valid).astype(np.int64)
            #src_y_valid = np.rint(src_y_valid).astype(np.int64)
            dst[layers_valid, dst_y_valid, dst_x_valid] = src[layers_valid, src_y_valid, src_x_valid]
        
        valid_mask = np.zeros([T, sH, sW])
        valid_mask[layers_valid,dst_y_valid, dst_x_valid] = 255

        return dst, valid_mask
    
    def generate_rotation_matrices(self, T, rotation_span, axis=None):
        """

        :param T: number of frames
        :param rotation_span: max rotation in radians - if radius = 0.5,
        angles are drawn from a uniform distribution over (-0.25, 0.25)
        :param axis: 0 = X Axis (A),1 = Y Axis (B), 2 = X Axis (C)
        :return: np.float64 array of shape T x 3 - A, B, C at each time step, t from 0 to T.
        """
        assert T> 0, 'Time frames must be > 0'
        assert 0<=rotation_span<=np.pi, 'Rotation radius must be in (0, pi)'

        rotations = np.zeros([T,3])
        if axis is not None:
            rotations[:, axis] = np.random.uniform(-rotation_span/2.0, rotation_span/2.0, T)
        elif axis is None:
            rotations = np.random.uniform(-rotation_span/2.0, rotation_span/2.0, (T,3))
        return rotations

    def generate_sequence(self, points, theta, dtheta, axis=0, max_rotation=np.pi/3):
        """
        Generate a sequence of points by rotating points sequentially
        from -theta to theta, over the specified axis

        :param T: number of frames
        :param points: K correspondences in the frame - shape = [1, K, 2]
        0 = x axis (horizontal)
        1 = y axis (vertical)

        :param theta: rotation range between 0 and np.pi
        :param dtheta: step size for rotation
        :param f_pixels: focal length in pixels, not used in the code
        :param axis: axis of rotation
        :param max_rotation: max allowed rotation
        :return: rotated correspondences -np.float64 array of shape [T,K, 2]
        """
        # rotation in radians
        T, K, _ = points.shape
        new_points = np.zeros([T,K,3])

        current_angle = theta

        points = points.reshape([1,K,3])

        initial_transform = np.zeros([1,6])
        initial_transform[0, axis] = current_angle
        current_points = apply_camera_transform(initial_transform, points)
        transform= np.zeros([1,6])
        transform[0, axis] = dtheta
        angles = []

        for i in range(T):
            new_points[i, :,: ]= current_points
            current_points = apply_camera_transform(transform, current_points)
            current_angle += dtheta
            angles.append(current_angle)
            if current_angle>=max_rotation:
                current_angle = theta
                new_transform = np.zeros([1,6])
                new_transform[0,axis] = current_angle
                current_points = apply_camera_transform(new_transform, points)
        return new_points

    def update_stacks(self, transformations, src_images, iteration=None):
        """
        update the pixel stack by copying pixels from the source image

        :param transformations: the transformation parameters, [T,6]
        :param src_images: the source images, [T, H, W, C]
        :param iteration: the iteration times for each parameter, defult None means no sequential transformations, [1, 6]

        :return: updated stack, [T, sH, sW, C]
        :return: masks, [T, sH, sW]
        """
        #iteration = np.floor(np.amax(np.absolute(transform[0:3]))*self.f_pixels).astype(int)
        T, _ = transformations.shape
        C = src_images.shape[3]
        assert src_images.shape[0] == T, 'invalid number of input source image'

        transforms = -np.copy(transformation)
        src_points = self.toolbox.apply_camera_transform(transforms, np.repeat(self.seed_points_norm, T, axis=0), iteration)
        src_points = self.denormalize_input(src_points)
        src = block_reduce(src_images, block_size=(1, self.sample, self.sample, 1), func=np.mean)
        dst = np.zeros((T, self.sH, self.sW, C))
        dst_images, masks = self.backward_transform(src, dst, src_points, np.repeat(self.seed_points, T, axis=0))

        return dst_images, masks
