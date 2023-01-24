# -*- coding: utf-8 -*-
"""
In this module, the balck borders around the images are removed, and the images are saved in the output directory.
The module uses multiprocessing, however if the code runs on few images it may be faster to use a single process.
"""

import multiprocessing

from . import black_border
from Dataset import dataset

def run_preprocessing(indices):
    inp = {i:dataset.indices[i]['path'] for i in indices}.items()
    if not dataset.INFERENCE_MODE:
        n_cpu = multiprocessing.cpu_count()
        print('Starting',n_cpu,'processes')
        p =  multiprocessing.Pool(n_cpu)
        return p.map(black_border.process, inp)
    return [*map(black_border.process, inp)]
        
def main(indices):

    preprocessing_summary = dataset.summaries.load('data', 'preprocessing')
    
    res = run_preprocessing(indices)
        
    preprocessing_summary = {**preprocessing_summary, **{index :summ for index, summ in res}}

    dataset.summaries.save(preprocessing_summary, 'data', 'preprocessing')

