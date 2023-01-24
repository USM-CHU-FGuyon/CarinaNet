# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:36:43 2022

@author: 151985
"""

import matplotlib.pyplot as plt

from Dataset import dataset

def _plot_annot(index, data_summary):
    img = dataset.load.compressed_image(index)
    annot = data_summary[index]['annotations']
    
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.scatter(*annot['CARINA'], c='b', alpha=0.5)
    ax.scatter(*annot['ETT'],c='r',alpha=0.5)
    for apex in annot['APEX']:
        ax.scatter(*apex, c='c', alpha=0.5)
    plt.title('Ground truth on '+index)
    path = dataset.paths.annot_visu(index)
    dataset.save.savefig(fig, path)
    print(f'   -> saved {dataset.paths.printable(path)}')
    plt.close()


def main(indices):
    if not dataset.annoted:
        return
    data_summary = dataset.summaries.load('data', 'image')
    try : 
        for index in indices:
            if dataset.annot.is_annoted(index):
                _plot_annot(index, data_summary)
    except KeyboardInterrupt :
        print('Keyboard Interrupted visualize...')