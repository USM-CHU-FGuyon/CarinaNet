# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:24:45 2022

@author: 151985
"""

from . import json_utils

"""
Class for handling summaries at each phase and step of the process. 

A sumary is a dict, saved as a json file that has keys '1', '2', '3'... corresponding 
to each image as specified in Dataset.indices. Values depend on the process that is applied.

Each summary is associated with a ```step``` and a ```phase```, 
and will be saved under ```outputs/{phase}/{step}_summary.json```

"""


class Summaries(object):
    def __init__(self, PathHandler):
        self._paths = PathHandler
        self._cached_summaries = {}
        
    def _key(self, phase, step):
        return f'{phase}_{step}'
    
    def load(self, phase, step, strict = False):
        """
        Returns the summary without loading the json file if it is already in _cached_summaries. 
        Else, loads the json file and adds the summary to the _cached_summaries.
        if strict is True, the method raises an error if the summary is empty.
        
        """
        try : 
            summ = self._cached_summaries[self._key(phase, step)]
        except KeyError:
            self._cached_summaries[self._key(phase, step)] = json_utils.loadjson(self._paths.summary(phase, step))
            summ = self._cached_summaries[self._key(phase, step)]
        if strict and not summ:
            raise ValueError(f'Summary {phase}, {step} is empty.')
        return summ
        
    def save(self, summary, phase, step):
        """
        Saves the summary as a json file and stores the dictionnary in the _cached_summaries dict.
        """
        json_utils.dumpjson(summary, self._paths.summary(phase, step))
        self._cached_summaries[self._key(phase, step)] = summary
    
        