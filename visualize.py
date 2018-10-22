#  Copyright 2018 Computer Vision Lab, UMass Amherst. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
""".visualize visualize the """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import sys
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

import seaborn as sns
import imageio

class Visualize():
    """Visualize class with visualization methods for camera rotation estimation."""

    def __init__(self, output_dir=None):
#    """These are the parameters that work for visualization.
#
#    Args:
#        output_dir: string
#            The default output directory for all the plots.
#    """
        #print("done")
        self.output_dir = output_dir

    def get_heatmap(self, data, axis=False, cmap=None, annot=None, fmt='.2g', output_dir=None, name=None):
#    """Plot the heatmap as a color-encoded matrix for  rectangular data.
#
#    Args:
#        data: ndarray rectangular data with shape (H, K)
#            Entropy or Likelihood data for all pixels in the image.
#        axis: bool, optional
#            The parameter controlling to show axis or not
#        cmap: matplotlib colormap name or object, or list of colors, optional
#            The mapping from data values to color space.
#        annot:  bool, optional
#            If True, write the data value in each cell.
#        fmt: string, optional
#            String formatting code to use when adding annotations.
#        output_dir: string
#            The output directory for the plots.
#    """
        print('heatmap')

        if output_dir is None:
            output_dir = self.output_dir
            
        H,W = data.shape

        if axis is False:
            plt.axis('off')
        sns.set()
        ax = sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt)
        if name is None:
            plt.savefig(output_dir + 'heatmap.png')
        else:
            plt.savefig(output_dir + name)
        #plt.show()

    def get_movie(self, indir, fmt='png', exportname=None, output_dir=None):
#    """Film the output images of congealing as a movie.
#
#    Args:
#        indir: string
#            The directory path where all output images are stored.
#        exportname: string
#            The mapping from data values to color space.
#        annot:  bool, optional
#            If True, write the data value in each cell.
#        fmt: string, optional
#            String formatting code to use when adding annotations.
#        output_dir: string
#            The output directory for the plots.
#    """
        print("Generating Movie")

        if output_dir is None:
            output_dir = self.output_dir

        frames = []
        filenames = glob.glob(indir+'/*.' + fmt)
        filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        # Load each file into a list
        #print filenames
        for filename in filenames:
            frames.append(imageio.imread(filename))

        # Save them as frames into a gif
        if exportname is None:
            exportname = "congealing.mp4"

        kargs = { 'fps': 30 }

        imageio.mimsave(output_dir + exportname, frames, 'mp4', **kargs)


    def key_points_plots(self, head, tail, height, width, axis='on', output_dir=None):
#    """Get the plots of the positions for corresponding key points across frames 
#       in one figure before and after congealing.
#
#    Args:
#        head: ndarray
#            The key points position across frames before congealing.
#        tail: ndarray
#            The key points position across frames after congealing.
#        height: The height of the original image.
#        width: The width of the original image.
#        axis: string 'on' or 'off', optional
#            Control axis and grid.
#        output_dir: string
#            The output directory for the plots.
#    """
        print("key_points_plots")
        if output_dir is None:
            output_dir = self.output_dir
            
        colour_seq= ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
        T,K,_ = head.shape

        fig = plt.figure(1)
        plt.xlim(0,width)
        plt.ylim(height,0)
        if axis is 'on':
            plt.axis('on')
            plt.grid(True)
        elif axis is 'off':
            plt.axis('off')
        else:
            raise ValueError("Argument for axis control must be 'on' or 'off'.")
            return
        for i in range(T):
            plt.scatter(x = head[i,:,0], y = head[i,:,1], s = 1, c =colour_seq, marker='o')
        plt.savefig(output_dir + "head.png")

        T,K,_ = tail.shape
        fig = plt.figure(2)
        plt.xlim(0,width)
        plt.ylim(height,0)
        if axis is 'on':
            plt.axis('on')
            plt.grid(True)
        elif axis is 'off':
            plt.axis('off')
        else:
            raise ValueError("Argument for axis control must be 'on' or 'off'.")
            return
        for i in range(T):
            plt.scatter(x = tail[i,:,0], y = tail[i,:,1], s = 1, c =colour_seq, marker='o')
        plt.savefig(output_dir + "tail.png")

    def key_points_movie(self, key_points, height, width, name=None, output_dir=None):
#        """Plot key points of each frame and make a movie."""
#
#    Args:
#        key_points: The key points position across frames. A numpy array of shape (T,K,2).
#        height: The height of the original image.
#        width: The width of the original image.
#        output_dir: string
#            The output directory for the plots.
#    """
        print("key_points_movie")
    
        if output_dir is None:
            output_dir = self.output_dir
            
        T,K,_ = key_points.shape
        colour_seq= ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        result = key_points[0]
        fig = plt.figure()
        ax = plt.axes(xlim=(0, width), ylim=(0, height))
        scat = ax.scatter(result[:,0], result[:,1], s=10, c=colour_seq, marker='o')

        # animation function.  This is called sequentially
        def animate(i):
            result = key_points[i+1]
            scat.set_offsets(result)

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, frames=T-1, interval=5)

        anim.save(output_dir+'key points.mp4', fps=10)

def main():
    print('main')
    pass


if __name__ == '__main__':
    main()

