# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:08:03 2022

@author: 151985
"""

import numpy as np
from Dataset import dataset


def compute(err_dic):    

    new_names = {'Barely readable': 'Barely readable',
                 'Low': 'Hardly visible carina',
                 'Medium':'Poor quality',
                 'Normal':'Acceptable quality',
                 'Unspecified':'Unspecified'}    

    qual_err, qual_std = {}, {}   

    data = {key : {lab: [] for lab in dataset.annot.quality_label.values()} for key in err_dic}
    
    for key in err_dic:
        print(f'\n{key}')
        for index, err in err_dic[key].items():
            data[key][dataset.annot.quality(index)].append(err)
    
        for quality, errors in data[key].items():
            if len(errors) == 0:
                continue
            qual_err[quality] = np.mean(np.abs(errors))
            qual_std[quality] = np.std(errors)
            
            print(f'{new_names[quality]}'
                  f'    {qual_err[quality]:.2f}cm, '
                  f'std : {qual_std[quality]:.2f}, '
                  f'n = {len(errors)}'
                  )
    