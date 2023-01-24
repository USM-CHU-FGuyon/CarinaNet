import numpy as np
from pathlib import Path
import os
import collections

class Metrics(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self._to_cm = dataset.pixel_spacing
        self._path_to_pixel_spacing = dataset.path_to_pixel_spacing

        self.pixel_spacing = self._build_pixel_spacing(dataset.fnames)

    def _build_pixel_spacing(self, fnames):
        if self._path_to_pixel_spacing == '': #constant pixel spacing for all images
            return {f : self._to_cm for f in fnames}
        pixel_spacing_csv = np.genfromtxt(self._path_to_pixel_spacing, dtype=str, delimiter = ',')

        try :
            
            fnames_to_spacing = {(Path(self.dataset.paths.db_path) / fname).as_posix() : float(spacing) for fname, spacing in pixel_spacing_csv}
            if collections.Counter(fnames) != collections.Counter(fnames_to_spacing.keys()):
                print(fnames[0])
                print([*fnames_to_spacing.keys()][0])
                
                
                raise ValueError(f'Content of {self.dataset.paths.db_path} does not match files listed in {self._path_to_pixel_spacing}')
            return fnames_to_spacing
        except ValueError :
            print(f'Encontered error while reading {self._path_to_pixel_spacing}')
            raise
    def err1d(self, GT, pos, index):
        return (GT[1]-pos[1])*self.to_cm(index)

    def uncertainty(self, confidence):
        return 3.9*np.exp(-2.1*confidence) - 0.3

    def to_cm(self, index):
        return self.dataset.indices[index]['pixel_spacing']/10
