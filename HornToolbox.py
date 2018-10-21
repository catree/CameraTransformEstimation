import numpy as np
import sys

class HornToolbox:

    def __init__(self):
        """
        Constructor
        """
        pass

    def unit_transform(self, transformations, reference_points, return_flow=False):
        """
        compute the 3D flow and positions after transformation.

        :param transformations: camera transformation parameters, [T, 6]
        :param reference_points: the normalized location of pixels, [T, H*W, 3]
        :param return_flow: control the output to be flow or transformed location, boolean, default=flase (option)

        :return flow or pixel positions
        """
        assert transformations.shape[0]==reference_points.shape[0], 'Invalid shapes in apply_camera_transform'

        T, K, _ = reference_points.shape

        flow = np.zeros_like(reference_points)

        x = reference_points[:,:,0]
        y = reference_points[:,:,1]
        z = reference_points[:,:,2]

        A = transformations[:,0]
        B = transformations[:,1]
        C = transformations[:,2]
        U = transformations[:,3]
        V = transformations[:,4]
        W = transformations[:,5]

        flow[:, :, 0] = A * (x * y) - B * (x ** 2 + 1) + C * y + Z * (-U + W * x)
        flow[:, :, 1] = A * (y ** 2 + 1) - B * (x * y) - C * x + Z * (-V + W * y)

        if return_flow:
            return flow
        else:
            return reference_points + flow

    def apply_camera_transform(self, transformations, reference_points, iteration=None, return_flow=False):
        """
        apply sequential transform if the transformation parameters are large.

        :param transformations: camera transformation parameters, [T, 6]
        :param reference_points: the normalized location of pixels, [T, H*W, 3]
        :param iteration: the minimum transform steps for rotations and translations, [1, 6]
        :param return_flow: control the output to be flow or transformed location, boolean, default=flase (option) 

        :return: rotated normalized points
        """
        if iteration is None:
            return self.unit_transform(transformations, reference, return_flow)
        
        iteration = np.amin([9, iteration]).astype(int) + 1

        dTransforms= transformations / iteration

        T, K, _ = reference_points.shape
        output_points = np.copy(reference_points)

        for j in range(iteration):
            output_points = self.unit_transform(dTransforms, output_points, return_flow)

        return output_points

    def compute_rotations(self, reference_points, data):
        """
        given target location of pixels and current location of pixels, get the estimated rotations

        :param reference_points: the normalized target location to align all data to, [T, K, 2]
        :param data: the normalized original location of points, [T, K, 2]
        :return: A, B, C params that aligns each frame of points to reference_points. [T,3]
        """
        assert reference_points.shape == data.shape, 'Invalid shapes in compute_rotations'

        frames = np.copy(data)

        T, K, _ = frames.shape

        M_matrices = np.zeros([T, 3, 3])
        n_vectors = np.zeros([T, 3])

        x = frames[:, :, 0]
        y = frames[:, :, 1]

        flow = reference_points - frames

        u = flow[:, :, 0]
        v = flow[:, :, 1]

        x_sq = x ** 2
        y_sq = y ** 2

        a = np.sum(x_sq * y_sq + (y_sq + 1) ** 2, axis=1)

        b = np.sum((x_sq + 1) ** 2 + x_sq * y_sq, axis=1)

        c = np.sum(x_sq + y_sq, axis=1)

        d = np.sum(-x * y * (x_sq + y_sq + 2), axis=1)

        e = np.sum(-y, axis=1)

        f = np.sum(-x, axis=1)

        M_matrices[:, 0, 0] = a
        M_matrices[:, 0, 1] = d
        M_matrices[:, 0, 2] = f
        M_matrices[:, 1, 0] = d
        M_matrices[:, 1, 1] = b
        M_matrices[:, 1, 2] = e
        M_matrices[:, 2, 0] = f
        M_matrices[:, 2, 1] = e
        M_matrices[:, 2, 2] = c

        k = np.sum(u * x * y + v * (y_sq + 1), axis=1)
        l = np.sum(-(u * (x_sq + 1) + v * x * y), axis=1)
        m = np.sum((u * y - v * x), axis=1)

        n_vectors[:, 0] = k
        n_vectors[:, 1] = l
        n_vectors[:, 2] = m

        rotations = np.zeros([T, 3])

        M = np.linalg.inv(M_matrices)  # broadcastable!?

        for i in range(T):
            n = n_vectors[i, :]
            rotations[i, :] = M[i,...].dot(n)

        return rotations
