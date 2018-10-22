import os, glob, logging, time
import numpy as np
from skimage import io
import pandas as pd
import pickle
from CameraTransformUtils import CTUtils
from timer import Timer
from DatasetUtils import *
import matplotlib.pyplot as plt

from CoordinateCongealingToolbox import CCToolbox
from PixelCongealingToolbox import PCToolbox

log = logging.getLogger(__name__)


class CameraTransformEstimator:

    def __init__(self, config, sequence='goat', focal_length=None, coord_congeal=False, pixel_congeal=False):
        self.config = config
        self.coord_congeal = coord_congeal
        self.pixel_congeal = pixel_congeal
        log.info('Coord Congealing %s'% self.coord_congeal)
        log.info('Pixel Congealing %s'% self.pixel_congeal)

        self.input_directory = self.config['DATASET']['IMAGES'] + sequence
        self.output_directory = self.config['OUTPUT']['DIR'] + sequence
        self.fmt = self.config['DATASET']['FORMAT']
        self.cc_iters = int(self.config['COORD CONGEAL']['ITERS'])

        T = int(self.config['MISC']['T'])
        H = int(self.config['MISC']['H'])
        W = int(self.config['MISC']['W'])
        C = int(self.config['MISC']['C'])

        self.T = T
        self.H = H
        self.W = W
        self.C = C

        f_pixels = focal_length
        self.panH = int(1.5*H)
        self.panW = int(1.5*W)

        """ load tracks """
        tracks_location = self.config['DATASET']['TRACKS']
        tracks = self.load_tracks(tracks_location) + np.array([int(W/2.0),int(H/2.0)])
        t = tracks.shape[0]
        K = tracks.shape[1]
        self.tracks = np.ones((t,K,3))
        self.tracks[:,:,0:2] = tracks[:,:,:]

        assert self.tracks.shape[0]==T, 'Missing tracks'

        """ make sure output directory exists."""
        self.check_output_directory()

        log.info('Using H = %s, W=%s, C=%s, f_pixels=%s, T= %s'%(H, W, C, f_pixels, T))
        utils = CTUtils(H, W, f_pixels, self.panH, self.panW, toolbox='Horn')
        self.CCT = CCToolbox(H, W, C, T, K, f_pixels, utils)
        self.PCT = PCToolbox(H, W, C, T, f_pixels, self.panH, self.panW, utils)
        self.image_stacks = np.zeros((T, self.panH, self.panW, C))
        self.mask_stacks = np.zeros((T, self.panH, self.panW, 1))

    def get_average_image(self, pixel_stacks):
        nonzeros = np.count_nonzero(np.sum(pixel_stacks,axis=3),axis=0)
        nonzeros[nonzeros==0]=1
        nonzeros = np.repeat(nonzeros[...,np.newaxis],self.C,axis=2)
        mean_img = np.sum(pixel_stacks,axis=0)/nonzeros

        return mean_img

    def calculate_camera_params(self):
        self.params = np.zeros([self.T, 6])
        self.inv_dep_map = np.ones((self.panH, self.panW))
        assert self.CCT.T == self.PCT.T, "PCT and CCT mismatch in number of frames."
        if self.coord_congeal:
            self.params = self.perform_coord_congealing()

        if self.pixel_congeal:
            self.params, self.inv_dep_map = self.perform_pixel_congealing()
        #self.params = params

        return self.params, self.inv_dep_map

    def perform_coord_congealing(self):
        cc_timer = Timer()
        cc_timer.tic()
        log.info('Started co-ordinate congealing')
        #params, _ = self.CCT.successive_congealing(self.tracks, self.cc_iters, reference=0)
        params = self.CCT.successive_congealing(self.tracks, self.cc_iters)
        np.save(self.output_directory + '/CC Params.npy', params)
        self.params = params
        pixel_stacks = load_images(self.input_directory+ '/images/', self.config, filetype=self.fmt)

        CC_stacks = np.zeros((self.T,self.panH,self.panW,self.C))
        CC_stacks = self.CCT.utils.update_stacks(pixel_stacks, params, CC_stacks)
        #average_image = np.mean(CC_stacks, axis=0)
        average_image = self.get_average_image(CC_stacks)
        io.imsave(self.output_directory + '/AfterCC.png', average_image.astype(np.uint8))
        log.info('Completed coordinate congealing')
        cc_span = pc_timer.tic()
        log.info('Coordinate congealing running time:: ' + str(cc_span) + ' seconds.')
        return params

    def perform_pixel_congealing(self):
        os.makedirs(os.path.join(self.output_directory, '/PC_process'))
        os.makedirs(os.path.join(self.output_directory, '/PC_Params'))
        pc_timer = Timer()
        pc_timer.tic()
        log.info('Started Pixel Congealing.')
        print self.input_directory + '/images/'
        self.gray_stacks = load_images(self.input_directory + '/images/', self.config, filetype=self.fmt, grayscale_only=True)
        #params = np.zeros([self.PCT.T, 3])
        params = self.params
        init_level = int(self.config['PIXEL CONGEAL']['LEVEL'])
        inv_dep_map = self.inv_dep_map
        inv_dep_map = self.PCT.utils.update_inv_dep_map(inv_dep_map)

        sample = 5
        level = init_level
        gray_stacks_5 = block_reduce(self.gray_stacks, block_size=(1, self.sample, self.sample, 1), func=np.mean)
        params, inv_dep_map, history = self.PCT.coordinate_descent(params, gray_stacks_5, self.config, level, sample, self.output_directory)

        sample = 3
        level = int(init_level/2.0)
        gray_stacks_3 = block_reduce(self.gray_stacks, block_size=(1, self.sample, self.sample, 1), func=np.mean)
        params, inv_dep_map, history = self.PCT.coordinate_descent(params, gray_stacks_3, self.config, level, sample, self.output_directory)

        sample = 1
        level = int(init_level/4.0)
        gray_stacks_1 = block_reduce(self.gray_stacks, block_size=(1, self.sample, self.sample, 1), func=np.mean)
        params, inv_dep_map, history = self.PCT.coordinate_descent(params, gray_stacks_1, self.config, level, sample, self.output_directory)

        np.save(self.output_directory + '/PC Params.npy', params)
        pixel_stacks = load_images(self.input_directory+ '/images/', self.config, filetype=self.fmt)

        self.PCT.utils.pick_sample(1)
        inv_dep_map = self.PCT.utils.update_inv_dep_map(inv_dep_map)
        self.PCT.utils.pick_sample(1)
        PC_stacks = np.zeros((self.T,self.panH,self.panW,self.C))
        PC_stacks = self.PCT.utils.update_stacks(pixel_stacks, params, PC_stacks)
        #average_image = np.mean(pixel_stacks, axis=0)
        average_image = self.get_average_image(PC_stacks)
        io.imsave(self.output_directory + '/AfterPC.png', average_image.astype(np.uint8))

        with open(self.output_directory + '/history.p', 'wb') as f:
            pickle.dump(history, f)
        pc_span = pc_timer.tic()
        # plt.plot(history)
        # plt.xlabel('Iterations')
        # plt.ylabel('Entropy for Pixel Congealing')
        # plt.savefig(self.output_directory + '/Entropy.png')
        log.info('Completed pixel congealing.')
        log.info('Pixel congealing running time:: ' + str(pc_span) + ' seconds.')

        return params, inv_dep_map

    def check_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            os.mkdir(self.output_directory + '/images/')
            os.mkdir(self.output_directory + '/masks/')
            log.info('Output directory does not exist. Created.')

    def load_tracks(self, tracks_location):
        if self.config['MISC']['MANUALTRACKS'] =='TRUE':
            points = np.load(tracks_location)
        elif self.config['MISC']['MANUALTRACKS'] =='FALSE':
            numTracks = int(self.config['MISC']['K'])
            numPics = int(self.config['MISC']['T'])
            points = pd.read_table(tracks_location, sep="\s+", names=["Col1", "Col2", "Col3"])
            temp = points.loc[points['Col3'].isnull()]
            temp = temp[temp['Col1'] == 0.0]
            indices = temp[temp['Col2'] == float(numPics)].sample(numTracks).index.values
            indices = np.sum([np.repeat(indices, numPics), np.tile(np.arange(1, numPics + 1), numTracks)], axis=0)
            points = points.values[indices]
            points = points.reshape(numTracks, numPics, 3)[:, :, 0:2]
            points = np.swapaxes(points, 0, 1)
        else:
            print 'Bad config for MISC:MANUALTRACKS in config.ini'
            raise Exception
        return points

    def generate_aligned_images(self):
        os.makedirs(os.path.join(self.output_directory, '/images'))
        os.makedirs(os.path.join(self.output_directory, '/masks'))
        source_images_list = glob.glob(self.input_directory + '/images/*.'+self.fmt)
        source_images_list.sort()

        log.info('Generating the aligned images.')

        assert len(source_images_list)==self.params.shape[0], "Params and images don't match"

        log.info('Reading '+str(len(source_images_list))+' images from '+self.input_directory)
        self.PCT.utils.update_inv_dep_map(self.inv_dep_map)
        self.PCT.utils.pick_sample(1)

        src_images = np.zeros((self.T, self.panH, self.panW, 3))
        for i in range(self.T):
            fname = source_images_list[i]
            camera_matrix = np.reshape(self.params[i, :], [1, 6])
            src_images[i] = io.imread(fname)
            
        new_images, masks = self.PCT.utils.update_stacks(self, params, src_images, iteration=10)
        for i in range(self.T):
            self.image_stacks[i] = new_image
            self.mask_stacks[i] = mask[:,:,np.newaxis]
            # io.imsave(self.output_directory + '/images/img_' + str(i).zfill(3) + '.png',
            #           new_image.astype(np.uint8) / 255.0)
            io.imsave(self.output_directory + '/images/img_' + str(i).zfill(3) + '.png',
                      new_image.astype(np.uint8))
            io.imsave(self.output_directory + '/masks/img_' + str(i).zfill(3) + '.png',
                      mask.astype(np.uint8))

        average_image = np.mean(self.image_stacks, axis=0)
        io.imsave(self.output_directory + '/recover.png', average_image.astype(np.uint8))
        overlap = np.repeat(np.all((self.mask_stacks>0), axis=0), 3, axis=2)
        io.imsave(self.output_directory + '/maskrecover.png', (average_image * overlap).astype(np.uint8))

