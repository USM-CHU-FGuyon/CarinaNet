# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:35:43 2022

@author: 151985
"""
import os
from pathlib import Path

def dataset_fnames(path):
    return [(Path(path) / fname).as_posix() for fname in os.listdir(path) if fname.endswith('.png') 
                                                       or fname.endswith('.jpg')]


