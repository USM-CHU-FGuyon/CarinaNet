import os

import numpy as np

from . import json_utils, initialize

from .PathHandler import PathHandler
from .DataLoader import DataLoader
from .DataSaver import DataSaver
from .Annotations import Annotations
from .Summaries import Summaries
from .Metrics import Metrics
from .Dashboard import Dashboard
from . import TRAI_ICU_utils, CUSTOM_utils



class Dataset(object):


    def __init__(self, name = '', path_to_img = '', path_to_pixel_spacing = '',pixel_to_mm = 0.2,
                 xls_annot_path = '', INFERENCE_MODE = False, annoted = True):

        self.pixel_spacing = pixel_to_mm
        if not INFERENCE_MODE:
            self.paths = PathHandler(name, path_to_img, xls_annot_path)
            self.name = name
            self.path_to_pixel_spacing = path_to_pixel_spacing
            self.fnames = self._dataset_fnames()

            self.metrics = Metrics(self)
            self.annoted = annoted

            initialize.mkdir_outputs(self.paths)
            self.load = DataLoader(self.paths)
            self.save = DataSaver(self.paths)
            self.indices = self._get_dataset_indices()
            self.summaries = Summaries(self.paths)
            self.annot = Annotations(self)

            self.dashboard = Dashboard(self)
            self.test_indices = self._hist_train()
            
        self.INFERENCE_MODE = INFERENCE_MODE
        
    def _hist_train(self):
        if self.name == 'TRAI_ICU':
            try :
                file = np.genfromtxt(f'{self.paths.hist_traindir}test_annots.csv',
                                     delimiter = ',', dtype = str)
                file = np.atleast_2d(file)
                return np.unique([os.path.basename(s)[:-4] for s in file[:,0]])
            except OSError :
                return []
        return [*self.indices.keys()]
    
    def _ignored_data(self):
        """
        Reads the file ```Dataset/{name}/ignore_from_dataset.json``` that should be formatted as : 
        ```
        {
            'path/to/image1.png': 'excluding because reason A',
            'path/to/image2.png': 'excluding because reason B'
         }
        ```
        and excludes from the dataset all the images specified in that file 
        Returns
        -------
        list of image paths that should be ignored from the dataset

        """
        return [*json_utils.loadjson(self.paths.metadata_path+'ignore_from_dataset.json').keys()]

    def _build_index(self, fnames):
        """
        Returns and saves a json that links every image file to an index

        Parameters
        ----------
        fnames : list of paths.
        spacings : dic with keys fnames that contains the pixel spacing of corresponding image
        Returns
        -------
        Dictionnary where keys are '1', '2', '3'... 
        and values are 'path/to/image1.png', 'path/to/image2.png', 'path/to/image3.png' TODO : update

        """


        ignored_data = self._ignored_data()
        dic = {str(index + 1): {'path':fname,
                                'pixel_spacing': self.metrics.pixel_spacing[fname]}
               for index, fname in enumerate(fnames) if not os.path.basename(fname) in ignored_data}
        json_utils.dumpjson(dic, self.paths.indices)
        return dic

    def _dataset_fnames(self):
        """
        Returns a list of all the images found in the directory ```self.paths.db_path```.
        
        Returns
        -------
        list of paths

        """
        if self.name == 'TRAI_ICU':
            return TRAI_ICU_utils.dataset_fnames(self.paths.db_path)
        else :
            return CUSTOM_utils.dataset_fnames(self.paths.db_path)

    def _get_dataset_indices(self):
        """
        Creates the index linking all images to an int dataset.

        Raises
        ------
        ValueError
            Raises a ValueError if no file was found in the image dataset directory.

        Returns
        -------
        dict
            Dictionnary containing indices '1', '2', '3'... 
            linked to image paths 'path/to/image1.png', 'path/to/image2.png', 'path/to/image3.png'...
        """
        fnames = self.fnames
        if not fnames:
            raise ValueError(f'No images found at {self.paths.db_path}')

        if self.name == 'TRAI_ICU':
            fnames = TRAI_ICU_utils.sort_fnames(fnames)
        return self._build_index(fnames)

        
    def init_inference(self, path_to_img):
        self.name = 'inference'
        self.annoted = False
        self.paths = PathHandler(self.name, path_to_img, '')
        self.path_to_pixel_spacing = ''
        self.fnames = self._dataset_fnames()
        self.metrics = Metrics(self)
        initialize.mkdir_outputs(self.paths)
        self.load = DataLoader(self.paths)
        self.save = DataSaver(self.paths)
        self.indices = self._get_dataset_indices()
        self.summaries = Summaries(self.paths)
        self.annot = Annotations(self)

        self.dashboard = Dashboard(self)
        self.test_indices = self._hist_train()

