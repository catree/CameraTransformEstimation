import numpy as np
import sys

class ApproToolbox(object):

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
        Z = reference_points[:,:,2]

        frames = np.ones((4,K))
        frames[0] = x[0]
        frames[1] = y[0]
        frames[3] = Z[0]

        A = transformations[:,0].flatten()
        B = transformations[:,1].flatten()
        C = transformations[:,2].flatten()
        U = transformations[:,3].flatten()
        V = transformations[:,4].flatten()
        W = transformations[:,5].flatten()

        xfrms=np.ones((3*T,4))
        xfrms[3*np.arange(0,T)+0,1]=-C  
        xfrms[3*np.arange(0,T)+1,0]=C  
        xfrms[3*np.arange(0,T)+2,0]=-B  
        xfrms[3*np.arange(0,T)+0,2]=B  
        xfrms[3*np.arange(0,T)+1,2]=-A  
        xfrms[3*np.arange(0,T)+2,1]=A  
        xfrms[3*np.arange(0,T)+0,3]=U       
        xfrms[3*np.arange(0,T)+1,3]=V
        xfrms[3*np.arange(0,T)+2,3]=W

        coords=xfrms.dot(frames)
    
        if return_flow:
            return flow
        else:
            return np.swapaxes(coords.reshape((T,3,K)),1,2)

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
        
        #iteration = np.amin([4, iteration]).astype(int) + 1

        dTransforms= transformations / iteration

        T, K, _ = reference_points.shape
        output_points = np.copy(reference_points)

        output_points = self.unit_transform(dTransforms, output_points, return_flow)

        return output_points
