import numpy as np
import pandas as pd
import cv2, skimage.transform
import matplotlib.pyplot as plt

from . import json_utils

"""
Class for loading the needed data, images...

"""


def _imread_greyscale(path):
    img = cv2.imread(path, 0)  # loading as greyscale image
    if img is None:
        raise FileNotFoundError(f'cv2 failed to read {path}')
    return img

class DataLoader(object):
    def __init__(self, PathHandler):
        self.dataset_meta = PathHandler.metadata_path
        self.paths = PathHandler

    def annot(self):
        """
        Returns the raw annotations - before preprocessing.
        These raw annotations are copied in the ```outputs/data/image_summary.json```
        """
        return json_utils.loadjson(self.paths.annotations)

    def status(self, phase, step=''):
        return json_utils.loadjson(self.paths.status(phase, step))

    def db_image(self, path):
        """Loads a database image - before preprocessing - given the path to the image."""
        return _imread_greyscale(path)

    def image(self, index):
        """Loads a preprocessed image given an index"""
        return _imread_greyscale(self.paths.image(index))

    def compressed_image(self, index):
        """Loads a compressed version of a preprocessed image given an index. Only used for visualization."""
        return _imread_greyscale(self.paths.compressed_image(index))

    def incorrect_annot(self):
        try :
            return np.genfromtxt(f'{self.dataset_meta}incorrect_annotations.txt', dtype=str)
        except OSError:
            return []

    def mask(self, index, shape=None):
        mask_256 = np.load(self.paths.mask(index), allow_pickle=True)
        if shape is None:
            return mask_256
        return skimage.transform.resize(mask_256, shape) > .5  # re-binarize after resizing

    def contours(self, index):
        return np.load(self.paths.contour(index), allow_pickle=True)

    def visu_contour(self, index):
        return plt.imread(self.paths.ls_visu(index))

    def roi(self, index, mode):
        return _imread_greyscale(self.paths.roi(index, mode))

    def visu_roi(self, index, mode):
        return plt.imread(self.paths.visu_roi(index, mode))

    def edges(self, index):
        return np.load(self.paths.edges(index))

# =============================================================================
#     def filtered_edges(self, index):
#         return np.load(self.paths.image_augment_edges(index))
# =============================================================================

    def clusters(self, index):
        return np.load(self.paths.clusters(index))

    def visu_localization(self, index):
        return plt.imread(self.paths.visu_localization(index))

    def template_matching_map(self, index):
        return np.load(self.paths.template_matching_map(index))

    def ETT_roi(self, index):
        return _imread_greyscale(self.paths.ETT_roi(index))
    
    def ETT_detection(self, index):
        return np.load(self.paths.ETT_detection(index))
    
    def image_augment_clusters(self, index):
        return np.load(self.paths.image_augment_clusters(index))

    def image_augment_edges(self, index):
        return cv2.imread(self.paths.image_augment_edges(index),0)

    def successful_indices(self, phase, step=''):
        # TODO: probably get rid of this.
        """
        returns the indices of images for which a phase was successful by reading the status.json file.

        Parameters
        ----------
        step : str,('lung_segmentation' 'roi' or 'probe'),
        name of the phase for which we want the successful indices

        Returns
        -------
        list of indices in which the phase in argument was achieved successfully

        """
        try:
            return sorted([i for i, val in self.status(phase, step).items() if val['success']], key=int)
        except KeyError as e:
            print(e)
        return []
    