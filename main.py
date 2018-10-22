import logging, time
import argparse
from CameraTransformEstimator import CameraTransformEstimator
from visualize import Visualize
from setup import *
from timer import Timer
import numpy as np

parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('Required named arguments')
requiredNamed.add_argument('-f', "--focal_length", type=float, help="Focal Length in Pixels",required=True)

parser.add_argument("--movie", help="Flag to generate mp4 movie", action="store_true")
parser.add_argument("--log", help='Logfile. filename.log', default="CameraTransformEstimator.log")
parser.add_argument("-c", "--config", help='Path to config.ini', default='config.ini')
parser.add_argument("-s", "--sequence", help='Sample sequence from Dataset. Default= goat', default='goat')
parser.add_argument("--cc", help="Perform coordinate congealing.", default=False, action="store_true")
parser.add_argument("--pc", help="Perform pixel congealing.", default=False, action="store_true")

args = parser.parse_args()

config_settings = read_config(args.config)

logging.basicConfig(filename=args.log, level=logging.INFO)

prep_dir()

glob_timer = Timer()
glob_timer.tic()

CTEstimator = CameraTransformEstimator(config_settings, args.sequence, args.focal_length, args.cc, args.pc)
output_dir = CTEstimator.output_directory
final_params, inv_dep_map = CTEstimator.calculate_camera_params()

CTEstimator.generate_aligned_images()
np.save(output_dir+'/final_params.npy', final_params)
np.save(output_dir+'/inv_dep_map.npy', inv_dep_map)

visualizer = Visualize(output_dir=output_dir+'/')
visualizer.get_heatmap(inv_dep_map, cmap=plt.get_cmap('Blues'), output_dir=output_dir+'/', name = 'inverse_depth_map.png')

if args.movie:
    visualizer.get_movie(indir=output_dir+'/images/', fmt='png', exportname=None, output_dir=output_dir+'/')
    if args.pc:
    	visualizer.get_movie(indir=output_dir+'/PC_process/', fmt='png', exportname='PC_process.mp4', output_dir=output_dir+'/')
    logging.info('Generated an mp4 movie.')

glob_span = glob_timer.toc()
logging.info('Total running time: Completed in ' + str(glob_span)+' seconds.')
