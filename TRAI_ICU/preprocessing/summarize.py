# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:34:57 2022

@author: 151985
"""

from Dataset import dataset

def _shift_pos(pos, shift):
    return [p - s for p, s in zip(pos, shift)]

def _get_repositionned_annot(index, preprocessing_summary):

    annotations = dataset.annot.raw_annot(index)

    shift = [x[0] for x in preprocessing_summary[index]['cropping']]
    annotations['ETT'] = _shift_pos(annotations['ETT'], shift)
    annotations['CARINA'] = _shift_pos(annotations['CARINA'], shift)
    annotations['APEX'] = [_shift_pos(apex, shift) for apex in annotations['APEX']]
    return annotations
    
def _summarize(index, pp_summary):
    return {
            'shape': pp_summary[index]['shape'],
            'annotations': _get_repositionned_annot(index, pp_summary)
            }

def main():
    print('   Summarize...')
    pp_summary = dataset.summaries.load('data', 'preprocessing')
    data_summary = {index : _summarize(index, pp_summary) for index in pp_summary}

    dataset.summaries.save(data_summary,'data','image')
    print('      -> Done')