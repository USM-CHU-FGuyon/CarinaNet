# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 13:35:45 2021

@author: 151985
"""

import pandas as pd
import numpy as np



def _read_annot(xls_annot_path):
    try : 
        recueil_xls = pd.read_excel(xls_annot_path)
    except FileNotFoundError :
        raise FileNotFoundError(f'Could not find xls_annot_path "{xls_annot_path}"')
    recueil_xls.columns = ['numéro radio','position',
                           'extremité sonde X', 'extremité sonde Y',
                           'coin inf gauche X','coin inf gauche Y',
                           'coin sup droit X','coin sup droit Y',
                           'apex droit X','apex droit Y',
                           'apex gauche X','apex gauche Y',
                           'carene X', 'carene Y',
                           'Qualité','Lecteur', 'QUALITE']
    recueil_xls = recueil_xls.drop(columns='QUALITE')#not an actual column
    recueil_xls = recueil_xls.drop(index=0)#column names in the excel file
    
    recueil_xls = recueil_xls.apply(pd.to_numeric, errors='coerce')#convert all to numeric
    recueil_xls['numéro radio'] =recueil_xls['numéro radio'].astype(str)#except the index
    recueil_xls.set_index('numéro radio', inplace=True)
    return recueil_xls


def get_annotations_as_dict(xls_annot_path, incorrect_annotations):
    annotation_df = _read_annot(xls_annot_path)
    
    annotation_df.loc[incorrect_annotations] = np.nan
    annotation_df['ETT'] = annotation_df[['extremité sonde X', 'extremité sonde Y']].apply(lambda x : [*x], axis=1)
    annotation_df['CARINA'] = annotation_df[['carene X', 'carene Y',]].apply(lambda x : [*x], axis=1)
    annotation_df['APEX'] = annotation_df[['apex droit X','apex droit Y', 'apex gauche X','apex gauche Y',]].apply(lambda x : [[*x[:2]],[*x[2:]]], axis=1)
    annotation_df['zone_labelisee'] = annotation_df[['coin inf gauche X','coin inf gauche Y', 'coin sup droit X','coin sup droit Y']].apply(lambda x : [[*x[:2]],[*x[2:]]], axis=1)

    annotation_df = annotation_df[['ETT','APEX','CARINA','Qualité','position']]
    
    annotation_df.columns = ['ETT', 'APEX', 'CARINA', 'qualite', 'position']
    return annotation_df.to_dict('index')


