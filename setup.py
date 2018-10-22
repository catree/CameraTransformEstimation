import numpy as np
import glob
import os
from setuptools import setup, find_packages
from skimage import io, color
import configparser
import logging

BASE_PATH = '../'
INPUT_PATH = os.path.join(BASE_PATH, 'Input/')
OUTPUT_PATH = os.path.join(BASE_PATH, 'Output/')

log = logging.getLogger(__name__)

def prep_dir():
    os.makedirs(os.path.join(INPUT_PATH, 'Data'))
    os.makedirs(os.path.join(INPUT_PATH, 'Tracks'))
    os.makedirs(os.path.join(INPUT_PATH, 'configs'))
    os.makedirs(os.path.join(INPUT_PATH, 'params'))
    os.makedirs(os.path.join(INPUT_PATH, 'scripts'))
    os.makedirs(os.path.join(OUTPUT_PATH, 'logs'))


def read_config(configpath=INPUT_PATH + 'configs/config.ini'):
    config = configparser.ConfigParser()
    config.read(configpath)
    return config


def load_images(image_directory, config, filetype='jpg', grayscale_only=False):
    images_list = glob.glob(image_directory+'/*.'+filetype)
    images_list.sort()
    assert len(images_list)!=0, 'No input image'
    log.info('Read '+ str(len(images_list)) + 'images from '+image_directory)
    H = int(config['MISC']['H'])
    W = int(config['MISC']['W'])
    C = int(config['MISC']['C'])
    T = int(config['MISC']['T'])
    if grayscale_only:
        images_stack = np.zeros([T, H, W, 1], dtype=np.uint8)
    else:
        images_stack = np.zeros([T, H, W, C], dtype=np.uint8)

    for i in range(T):
        img = io.imread(images_list[i])
        img = img.astype(np.uint8)
        if grayscale_only:
            gray = color.rgb2gray(img) * 255.0
            images_stack[i, ..., 0] = gray.astype(np.uint8)
        else:
            images_stack[i, ...] = img

    return images_stack

