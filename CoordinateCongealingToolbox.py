from HornToolbox import HornToolbox
import numpy as np

class CCToolbox:

    def __init__(self, H, W, C, T, K, f_pixels, utils):
        """
        :param H: height of the images
        :param W: width of the images
        :param C: number of channels
        :param T: number of frames
        :param K: number of tracks
        :param f_pixels: focal length in pixels
        """
        self.H = H
        self.W = W
        self.C = C
        self.T = T
        self.K = K
        self.f_pixels = f_pixels # focal length in pixels
        self.utils = utils
        self.HornToolbox = HornToolbox()

    def successive_congealing(self, data, n_iters=10, reference=None):
        """
        :param data: [T, K, 2] np.float64 array original (unnormalized) (0,H)(0,W)
        :param n_iters: number of steps of congealing to perform
        :param reference: reference frame to align data to. default = mean.
        value from 0 to T-1 otherwise. Align all frames with the target.
        :return: locations after congealing. (unnormalized) (0,H)(0,W)
        :return: rotations after congealing.
        """
        #print "data",data.shape,np.amin(data[:,:,0:2]),np.amax(data[:,:,0:2])
        current_points = self.utils.normalize_input(data)

        transformation = np.zeros([self.T, 6])

        for j in range(n_iters):
            if reference is None:
                target_location = np.mean(current_points, axis=0)[np.newaxis, ...]
            else:
                target_location = current_points[reference,...][np.newaxis, ...]
            estimated_rotations = self.HornToolbox.compute_rotations(target_location, current_points)
            transformation[:,0:3] += estimated_rotations
            estimated_locations = self.HornToolbox.apply_sequential_transform(transformation, current_points)
            #estimated_locations = self.HornToolbox.apply_camera_transform(estimated_rotations, current_points)
            current_points = estimated_locations

        #estimated_rotations = transformation[0:3]
        #current_points = self.utils.normalize_input(data)
        #estimated_locations = self.HornToolbox.apply_sequential_transform(estimated_rotations, current_points)
        return transformation#, estimated_locations

