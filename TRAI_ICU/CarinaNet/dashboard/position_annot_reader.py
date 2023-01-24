# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:18:34 2022

@author: 151985
"""
import matplotlib.pyplot as plt
import numpy as np
from Dataset import dataset


global colors
    
colors = {
        'Selective': 'indigo',
        'Too low': 'red',
        'Good position': 'green',
        'Too high': 'orange'
        }
new_labels = {
    'Selective': 'Bronchial insertion',
    'Too low':'Too low',
    'Good position':'Good',
    'Too high':'Too high'
}

def _plot_hists(dic):

    bins = np.linspace(-3, 10, 20)
    data = {}
    for label, val in dic.items() :
        data[label], _ = np.histogram(val, bins)
    del data['tracheo']

    fig, ax = plt.subplots()

    for label, val in data.items():
        ax.bar(bins[:-1], val, 0.5, label = new_labels[label], color = colors[label])

    ax.legend(prop={'size': 6})
    ax.set_xlabel('ETT-Carina distance (cm)')
    ax.set_ylabel('Number of chest radiographs')
    #plt.title('Annotated ETT-Carina distance (cm)')
    plt.show()
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}position_classification')
    ax.set_yscale('log')
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}position_classification_logscale')
    plt.show()

def plot(spacing):
    
    GT_positions = {}
    for label in map(dataset.annot.ETT_position_label, range(1,6)):
        GT_positions[label] = [val['GT'] for index, val in spacing.items() if dataset.annot.pos_label(index)==label]
   
    _plot_hists(GT_positions)
    