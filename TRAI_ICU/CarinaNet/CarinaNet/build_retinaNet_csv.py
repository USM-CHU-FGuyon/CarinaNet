"""
Build a csv with annotations for training retinaNet.

annotations format :
/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat
/data/imgs/img_002.jpg,22,5,89,84,bird
/data/imgs/img_003.jpg,,,,,


Class Mapping format
cow,0
cat,1
bird,2

FROM : https://github.com/yhenon/pytorch-retinanet
"""
import numpy as np
import random
from pathlib import Path

from Dataset import dataset

def _format_annot(element, index):
    """
    returns a bounding box around an element,
    ex : '/data/imgs/img_001.jpg,837,346,981,456,cow'
    """
    if element =='CARINA':
        xp,xm,yp,ym = [100,100,100,100]#square box
    elif element == 'ETT':
        xp,xm,yp,ym = [100,100,100,100]
    
    
    if element=='CARINA':
        annot = dataset.annot.carina_img(index)
    elif element=='ETT':
        annot = dataset.annot.probe_img(index)
    
    if not any([np.isnan(a) for a in annot]):
        pth = Path(dataset.paths.image(index)).resolve().as_posix()
        return f'{pth},{annot[0]-xm},{annot[1]-ym},{annot[0]+xp},{annot[1]+yp},{element}\n'
    return ''

def _save_file(fname, indices):
    with open(fname, 'w') as f:
        for index in indices:
            f.write(_format_annot('CARINA', index))
            f.write(_format_annot('ETT', index))
    print(f'   -> saved {fname}')

    

def build_class_mapping(classes = ['CARINA', 'ETT']):
    """
    Builds the class mapping file from a list of classes.
    """
    fname = dataset.paths.traindir+'class_list.csv'
    with open(fname, 'w') as f:
        for i, el in enumerate(classes[:-1]): 
            f.write(f'{el},{i}\n')
        f.write(f'{classes[-1]},{len(classes)-1}')
    print(f'   -> saved {dataset.paths.printable(fname)}')

def build_annot_files():
    
    train_sample = 0.8
    
    annoted_data_AND = dataset.annot.annoted_probe_AND_carina()
    annoted_data_OR = dataset.annot.annoted_probe_OR_carina()
    annoted_data_XOR = dataset.annot.annoted_probe_XOR_carina()
    
    test_ind = int(len(annoted_data_OR)*(1-train_sample)) 

    random.seed(123)
    random.shuffle(annoted_data_AND)
    
    
    test = annoted_data_AND[:test_ind]
    trainval = annoted_data_AND[test_ind:] + annoted_data_XOR
    random.shuffle(trainval)
    
    train = trainval[test_ind:]
    val   = trainval[:test_ind]
    
    print('\n\n     PREPARING TRAINING DATA')
    print(f'        Training : {len(train)} elements, \n        '
          f'Validation : {len(val)} elements,\n        '
          f'Testing : {len(test)} elements\n')
    
    path = dataset.paths.traindir
    fname_train, fname_val, fname_test = path+'train_annots.csv', path+'val_annots.csv' ,path+'test_annots.csv'

    _save_file(fname_train, train)
    _save_file(fname_val, val)
    _save_file(fname_test, test)

    return train, val, test
    
def build_inference_file(indices):
    fname = dataset.paths.traindir+'inference_indices.csv'
    with open(fname, 'w') as f:
        for index in indices:
            pth = dataset.paths.image(index)
            f.write(f'{pth},,,,,\n')
    print(f'        -> saved {dataset.paths.printable(fname)}')



    
    
    
    
    
    
    
    
    

