# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:35:43 2022

@author: 151985
"""
import os


def dataset_fnames(path):
    return [path + fname for fname in os.listdir(path) if fname.endswith('.jpg') ]

def sort_fnames(fnames):
    return sorted(fnames, key = lambda fname : int(os.path.basename(fname).split('.')[0]))
