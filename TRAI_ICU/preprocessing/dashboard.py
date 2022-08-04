# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:02:02 2022

@author: 151985
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

from Dataset import dataset

def _get_annot_indices(data_summary):
    try :
        elements = ['APEX', 'CARINA', 'ETT', 'position', 'qualite']
    except KeyError as e:
        print(' /!\ Failed to find annotations in data/image_summary.json, check that a correct path to annotations was given')
        raise e
    non_nan_indices = [[i for i, ds_i in data_summary.items() if not np.isnan(ds_i['annotations'][el]).any()] for el in elements]
    return dict(zip(elements, non_nan_indices))

def _piechart_annotations(annoted, n_tot):

    for obj, annot in annoted.items():
        dic = {
               'annoted': len(annot),
               'non_annoted':n_tot-len(annot)
               }

        dataset.dashboard.piechart(dic,
                       'Number of annoted '+obj+' out of ' +str(n_tot), 
                       phase = 'data'
                       )

def _piechart_quality(annoted, data_summary):

    dataset.dashboard.piechart_from_list([int(data_summary[index]['annotations']['qualite']) for index in annoted['qualite']],
                             title ='Annotated quality of '+str(len(data_summary)) +' images from the dataset',
                             phase = 'data',
                             display_labels= dataset.annot.quality_label)


def main():
    if not dataset.annoted:
        return
    print('    Making dashboard plots...')
    data_summary = dataset.summaries.load('data','image')
    
    annoted = _get_annot_indices(data_summary)

    if max([len(val) for val in annoted.values()]) >0:

        _piechart_annotations(annoted, len(data_summary))
        _piechart_quality(annoted, data_summary)


    print('   -> Done')
