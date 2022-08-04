# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:53:03 2022

@author: 151985
"""

import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


class Dashboard(object):
    def __init__(self, Dataset):
        self.dataset = Dataset
        
    def piechart(self, dic, title, phase):
        if sum([*dic.values()]) == 0: #empty piechart.
            print(f'No errors encountered in {phase} - {title}')
            return
        fig, ax = plt.subplots()
        ax.pie(dic.values(), labels = [str(k) for k in dic.keys()], autopct='%1.1f%%')
        ax.set_title(title)

        path = f'{self.dataset.paths.dashboard_dir(phase)}{title}.png'
        self.dataset.save.fig_and_pickle(fig, path, dpi = 500, bbox_inches='tight')

        plt.show()
        plt.close()

    def piechart_from_list(self, lst, title, phase, display_labels ={}):
        """
        Counts the occurence of each unique element from a list, and plots it in a piechart

        Parameters
        ----------
        lst : list, data that will be represented in the piechart
        title : string, title of the plot
        display_labels : dict, links values to a display name.
        Returns
        -------
        None.

        """
        if display_labels:
            self.piechart({display_labels[key]: val for key, val in Counter(lst).items()}, title, phase)
        else :
            self.piechart(dict(Counter(lst)), title, phase)


    def error_histogram(self, err_dic, phase, bins = 100, unit = '', note = ''):
        """Plots an error histogram from a dictionnary of the format : {index: float -> error}"""
        err_values = [float(e) for e in err_dic.values()]
        if all(np.isnan(err_values)):
            print('\n     /!\ All errors are nan, no error plot to show')
            return
        meanerr = np.nanmean(np.abs(err_values))
        medianerr = np.nanmedian(np.abs(err_values))
        fig, ax = plt.subplots()
        ax.hist(err_values, bins = bins)
        ax.set_title(f'Error of on {len(err_dic)} images')


        ax.annotate(f'mean err = {meanerr:.2f} {unit}\nmedian err = {medianerr:.2f} {unit}',
                    xy=(0.05, 0.75), xycoords='axes fraction',
                    bbox=dict(facecolor='none', edgecolor='k', pad=3))

        ax.set_xlabel('Error in cm')
        ax.set_ylabel('Number of images')
        path = self.dataset.paths.dashboard_dir(phase)+'hist_error'+note+'.png'

        self.dataset.save.fig_and_pickle(fig,path)

    
    