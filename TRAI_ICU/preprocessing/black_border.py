"""

This module allows to handle black borders in images.

"""
import numpy as np
from Dataset import dataset

def _crop(img,tol=25):
    """from https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy/132934"""
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    cropped = img[row_start:row_end,col_start:col_end]
    return cropped, ((int(col_start),int(col_end)), (int(row_start),int(row_end)))

def _get_summary(img, processed, cropping):
    return {'original_shape':img.shape[::-1],
            'cropping': cropping,
            'shape':processed.shape[::-1]}

def process(arg):
    index, path =  arg
    try : 
        img = dataset.load.db_image(path)
        cropped, cropping = _crop(img)
        dataset.save.save_preprocessed_image(cropped, index)
    except PermissionError:
        img, cropped, cropping = [np.zeros((1,1)), np.zeros((1,1)), [(0,1),(0,1)]]
    return index, _get_summary(img, cropped, cropping)

