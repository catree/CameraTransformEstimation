import numpy as np
import fast_histogram as fht

class EntropyEstimator:
    """An estimator class using histogram to estimate entropy."""

    def __init__(self):
        """
        Constructor
        """
        self.num_imgs = None
        self.shape = None
        self.num_bins = None
        self.hist = None
        self.val_range = None
        self.bin_size = None

    def build_histogram(self, pixel_stacks, val_range, nbins):
        """
        build up the histogram of each pixel stacks
        :param pixel_stacks: the pixel stacks need to build histogram - numpy array (T, H, W, C)
        :param val_range: a tuple of the range of pixel values
        :param nbins: number of bins
        :return: histogram of each position of pixel
        """
        # converting RGB to grayscale assuming 4th axis is channel
        if len(pixel_stacks.shape) > 3:
            imgs = np.mean(pixel_stacks, axis=3)
        else:
            imgs = pixel_stacks

        # Creating histogram for eaxh spatial pixel position flattened out
        T, H, W = imgs.shape
        self.num_imgs = T
        self.shape = H*W
        self.H = H
        self.W = W
        self.val_range = val_range
        self.num_bins = nbins
        self.bin_size = (val_range[1] - val_range[0]) / float(self.num_bins)
        imgs = imgs.transpose(1,2,0)
        imgs = imgs.reshape(H*W, -1)

        # initializing histogram and adding filler to avoid divide by zero error
        #hist = np.zeros((H*W, self.num_bins+1))
        hist = np.zeros((self.shape, self.num_bins+1))
        filler = np.ones((self.shape, self.num_bins)) #*1.0

        for k in range(0, H*W):
            hist[k, 0:self.num_bins] = (fht.histogram1d(imgs[k,:], range=val_range, bins=self.num_bins))

        hist[:, self.num_bins] = self.num_imgs
        #hist = hist/(self.num_imgs + self.num_bins)
        self.hist = hist
        self.filler = filler

        return hist

    def compute_entropy(self, percentage=None):
        """
        compute sum of entropy with a optional mask
        :param percentage: optional, portion of entropy values to use. None implies all.
        otherwise portion between 0 and 1
        :return sum_entropy: the sum of entropy
        """
        hist = np.amax(np.array([self.hist[:, 0:self.num_bins],self.filler]), axis=0)
        amount = np.repeat(np.sum(hist, axis=1).reshape((self.shape, 1)),self.num_bins,axis=1)
        weights = np.sum(self.hist[:, 0:self.num_bins], axis=1)/self.num_imgs
        hist = hist/amount
        entropy = - hist*np.log(hist)
        lin_entropy = np.sum(entropy, axis=1) * weights
        if percentage is None:
            avg_entropy = np.sum(lin_entropy)/np.sum(weights)
        else:
            assert 0<percentage<=1, 'Invalid percentage of entropy'
            sum_entropy = np.sort(lin_entropy)
            edge = int(sum_entropy.shape[0] * percentage)
            sum_entropy = sum_entropy[:edge]

        return avg_entropy, lin_entropy

    # def mask_compute_entropy(self, mask=None):
    #     """
    #     compute sum of entropy with a optional mask
    #     :param mask: optional, linear pixel mask of what part of histogram to use - numpy array (self.shape,)
    #     :return sum_entropy: the sum of entropy
    #     """
    #     assert mask.shape[0]==self.shape, 'Invalid shape of mask'
    #     entropy = - self.hist*np.log(self.hist)
    #     if mask is None:
    #         sum_entropy =  np.sum(entropy)
    #     else:
    #         sum_entropy =  np.sum(entropy[mask])
    #     return sum_entropy
     
    def get_updated_entropy(self, old_img, new_img, percentage=None):
        """
        update the entropy of when the old image changes to the new image
        :param old_img: the old image before rotation
        :param new_img: the new image after rotation
        :param percentage: optional, portion of entropy values to use.
        :return ent: updated entropy
        """
        if percentage is not None:
            assert 0<percentage<=1, 'Invalid percentage of entropy'

        H, W = (0,0)
        if len(new_img.shape) > 2:
            H, W, _ = old_img.shape
            old_img = np.mean(old_img, axis=2)
            new_img = np.mean(new_img, axis=2)
        else:
            H, W = old_img.shape
        assert H*W==self.shape, 'Invalid shape of image' 

        old_img = old_img.reshape(self.shape, -1)
        old_bin_id = self.find_bin(old_img)
        old_bin_id = old_bin_id.astype(int)


        new_img = new_img.reshape(self.shape, -1)
        new_bin_id = self.find_bin(new_img)
        new_bin_id = new_bin_id.astype(int)

        #/(self.num_imgs + self.num_bins)
        self.hist[np.arange(0, self.shape), old_bin_id] -= 1.0
        self.hist[np.arange(0, self.shape), new_bin_id] += 1.0

        ent, lin_entropy = self.compute_entropy(percentage)

        self.hist[np.arange(0, self.shape), old_bin_id] += 1.0
        self.hist[np.arange(0, self.shape), new_bin_id] -= 1.0

        return ent, lin_entropy


    def find_bin(self, value):
        """
        find where the value lands in the bins
        :param value: a double or a numpy array
        :return: a double or numpy array of the id of the value
        """
        binid = (value-self.val_range[0])/self.bin_size
        binid[binid<0] = self.num_bins
        binid[binid>=self.num_bins] = self.num_bins
        return np.squeeze(binid.astype(int), axis=1)
