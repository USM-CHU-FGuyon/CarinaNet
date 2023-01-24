import time

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from Dataset import dataset

def _detect_ridges(im, sigma, offset = 10):
    #M.Rudzki et al's Vessel Detection Method Based on Eigenvalues of the Hessian Matrix and its Applicability to Airway Tree Segmentation
    #https://stackoverflow.com/questions/48727914/how-to-use-ridge-detection-filter-in-opencv
    H = hessian_matrix(im, sigma)
    i1, i2 = hessian_matrix_eigvals(H)

    rdg = np.abs(i2)

    rdg[:,:offset] = 0
    rdg[:,-offset:] = 0
    rdg[:offset,:] = 0
    rdg[-offset:,:] = 0
    return rdg

def _process_ridges(edg):
    horizontal_kernel = np.array([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],#Now that is well coded
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1],
    ])
    edges = ndi.correlate(edg.astype(float), horizontal_kernel)
    edges = np.abs(edges)
    edges = skimage.morphology.area_opening(edges, area_threshold=1000).astype(np.float32)
    edges[:,:20] = 0
    edges[:,-20:] = 0

    
    return edges


def binarize_ridges(edges):
    return np.where(edges>np.quantile(edges, 0.95), 1, 0)#get top 5% activations

def run(indices):
    t0 = time.time()
    print('   RIDGE DETECTION')
    for index in indices:

        img = dataset.load.ETT_roi(index)

        rdg = _detect_ridges(img, sigma = 3)

        filtered_edg = _process_ridges(rdg)

        dataset.save.image_augment_ridges(rdg, index)
        dataset.save.image_augment_edges(filtered_edg, index)
    print(f'      -> Done in {time.time()-t0:.2f}s\n')