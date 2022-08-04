import os, pickle, cv2
import numpy as np
from PIL import Image
#from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt

from . import json_utils

"""
Class that does the saving of all useful files in the pipeline.
"""

class DataSaver(object):
    def __init__(self, PathHandler):
        self.paths = PathHandler

    def save_preprocessed_image(self, img, index):
        im = Image.fromarray(img)
        im.save(self.paths.image(index))
        im.save(self.paths.compressed_image(index),
                "JPEG", optimize=True, quality=30)

    def mask(self, mask, index):
        np.save(self.paths.mask(index), mask, allow_pickle=True)

    def contours(self, smooth_contours, index):
        np.save(self.paths.contour(index), np.array(smooth_contours, dtype=object), allow_pickle=True)

    def roi(self, roi, index, mode):
        Image.fromarray(roi).save(self.paths.roi(index, mode))

    def edges(self, edges, index):
        np.save(self.paths.edges(index), edges, allow_pickle=True)

    def filtered_edges(self, edges, index):
        np.save(self.paths.filtered_edges(index), edges, allow_pickle=True)

    def clusters(self, X, labels, index):
        labelled_points = np.column_stack((X, labels))
        np.save(self.paths.clusters(index), labelled_points)

    def template_matching_map(self, matching_map, index):
        np.save(self.paths.template_matching_map(index), matching_map)

    def status(self, status, phase, step=''):
        json_utils.dumpjson(status, self.paths.status(phase, step))

    def fig_and_pickle(self, fig, path, dpi = 300, bbox_inches = None):
        fig.savefig(path, dpi = dpi, bbox_inches = bbox_inches)
        pickle.dump(fig, open(path+'.pkl', 'wb'))
        fig.savefig(path+'.svg', bbox_inches = bbox_inches)
        print(f'   -> saved "{self.paths.printable(path)}" + pickle.')

    def savefig(self, fig, path):
        """
        Uses package moviepy to plot figures as jpg.
        This is faster than plt.savefig for saving compressed plots.

        Parameters
        ----------
        fig : matplotlib.figure.Figure object
            figure object that should be saved.
        path : str
            savepath.

        Returns
        -------
        None.

        """
        if not '.' in os.path.basename(path):
            print(f'Format not specified, saving as {path + ".jpg"}')
            path += '.jpg'
        #Image.fromarray(mplfig_to_npimage(fig)).save(path)
        fig.savefig(path)
    def img(self, arr, path):
        Image.fromarray(arr).save(path)

    def ETT_roi(self, roi, index):
        Image.fromarray(roi).save(self.paths.ETT_roi(index))

    def image_augment_ridges(self, rdg, index):
        plt.imsave(self.paths.image_augment_ridges(index), rdg )

    def image_augment_edges(self, edg, index):
        plt.imsave(self.paths.image_augment_edges(index), edg )

    def image_augment_clusters(self, cluster, index):
        np.save(self.paths.image_augment_clusters(index), cluster)
        
    def closest_cluster(self, clst, index):
        np.save(self.paths.image_augment_closest_cluster(index), clst)
        
    def ETT_detection(self, points, index):
        np.save(self.paths.ETT_detection(index), points)
    
    def edges_binary(self, edg, index):
        plt.imsave(self.paths.edges_binary(index), edg)