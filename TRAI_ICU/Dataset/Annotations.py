# -*- coding: utf-8 -*-
import numpy as np

from . import json_utils
from .TRAI_ICU import annot_utils

class Annotations(object):
    
    def __init__(self, Dataset):
        self.dataset = Dataset
        
        self._incorrect_annots = self.dataset.load.incorrect_annot()
        if Dataset.name == 'TRAI_ICU': #Name used in the internal dataset.
            self._xls_annot_to_json()
        
        self.raw_annots = Dataset.load.annot() if self.dataset.annoted else {}
        self.indices = Dataset.indices
        self._ETT_position_label = {np.nan: 'No annot',0: "Pas d'iot", 1: "Good position", 2: "Too high", 3: "Too low", 4: "Selective", 5: "tracheo"}
        self.quality_label = {0: "Barely readable", 1: "Low", 2: "Medium", 3: "Normal", 4:"Unspecified"}
        self._empty_annot = {"ETT": [np.nan, np.nan],"CARINA": [np.nan, np.nan],
                             "zone_labelisee": [[np.nan, np.nan],[np.nan, np.nan]],
                             "APEX": [[np.nan, np.nan],[np.nan, np.nan]],
                             "qualite": np.nan, "position": np.nan}


    def _data_summary(self):
        return self.dataset.summaries.load('data', 'image')
    
    def _xls_annot_to_json(self):
        annotations = annot_utils.get_annotations_as_dict(self.dataset.paths.xls_annot_path, self._incorrect_annots)
        json_utils.dumpjson(annotations, self.dataset.paths.annotations)
    
    def reload_annot(self):
        self.raw_annots = self.dataset.load.annot() if self.dataset.annoted else {}
    
    def raw_annot(self, key):
        if not self.dataset.annoted:
            return self._empty_annot
        try :
            return dict(self.raw_annots[self.dataset.indices[key]['path']])  # independant copy
        except KeyError : #On some development dataset the keys to annot were '1', '2', '3'
            try :
                return dict(self.raw_annots[key])  # independant copy
            except KeyError :
                return self._empty_annot
    
    def annot_img(self, index, el):
        """Returns the annotation of an element in the preprocessed image"""
        return self._data_summary()[index]['annotations'][el]   
 
    def image_shape(self, index):
        """Returns the shape of the preprocessed image"""
        return self._data_summary()[index]['shape']
    
    def carina_img(self, index):
        """Returns the coordinates of the carina for image 'index' in the coordinates of the preprocessed image"""
        return [np.nan if np.isnan(a) else int(a) for a in self._data_summary()[index]['annotations']['CARINA']]
    
    def probe_img(self, index):
        """Returns the coordinates of the probe for image 'index' in the coordinates of the preprocessed image"""
        return [np.nan if np.isnan(a) else int(a) for a in self._data_summary()[index]['annotations']['ETT']]
    
    def is_annoted(self, index, fields = ['APEX', 'CARINA', 'ETT']):
        """Checks if at least one of the fields is not NaN in the annotation"""
        if not index in self._data_summary():
            return False
        return any(map(lambda x: not np.isnan(self._data_summary()[index]['annotations'][x]).any(), fields))
    
    def probe_and_carina_are_annoted(self, index):
        """Checks if a specific index has annoted probe and carina"""
        return self.is_annoted(index, fields=['CARINA']) and self.is_annoted(index, fields=['ETT'])
    
    def probe_xor_carina_are_annoted(self, index):
        """Checks if a specific index has annoted probe and carina"""
        return self.is_annoted(index, fields=['CARINA']) != self.is_annoted(index, fields=['ETT'])
    
    def annoted_probe_AND_carina(self):
        """Returns all the indices where both the probe and the carina are annoted."""
        return [index for index in self.indices.keys() if self.probe_and_carina_are_annoted(index)]
    
    def annoted_probe_XOR_carina(self):
        """Returns all the indices where both the probe and the carina are annoted."""
        return [index for index in self.indices.keys() if self.probe_xor_carina_are_annoted(index)]

    def annoted_probe_OR_carina(self):
        """Returns all the indices where either the probe or the carina are annoted."""
        return [index for index in self.indices.keys() if self.is_annoted(index, fields = ['CARINA', 'ETT'])]
    
    def ETT_position_label(self, pos):
        return self._ETT_position_label[pos]
    
    def ETT_position(self, index):
        if np.isnan(self._data_summary()[index]['annotations']['position']):
            return np.nan
        return int(self._data_summary()[index]['annotations']['position'])

    def pos_label(self, index):
        return self.ETT_position_label(self.ETT_position(index))
    
    def quality(self, index):
        try : 
            return self.quality_label[self._data_summary()[index]['annotations']['qualite']]
        except KeyError:
            return "Unspecified"
    