import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import skimage.transform
from matplotlib.offsetbox import AnchoredText
import sys
sys.path.append('TRAI_ICU/CarinaNet/pytorch_retinanet/retinanet')

from .CarinaNet import build_retinaNet_csv
from .CarinaNet.pytorch_retinanet import inference

from Dataset import dataset

def _rescale_model_output(index, out_img, scale):
    """Removes the padding and rescales the output visu to the original shape"""
    original_shape = dataset.annot.image_shape(index)[::-1]
    xpad = int(32 - original_shape[0] * scale % 32)
    ypad = int(32 - original_shape[1] * scale % 32)

    depadded = out_img[:-xpad, :-ypad, 0]
    return skimage.transform.rescale(depadded, 1 / scale)


def _visu_CarinaNet(index, summ_i):
    
    
    err_carina = summ_i['CARINA']['err']
    err_probe = summ_i['ETT']['err']
    
    
    img = dataset.load.image(index)
    annot_carina = dataset.annot.carina_img(index)
    annot_probe = dataset.annot.probe_img(index)

    fig, ax = plt.subplots()
    c_color = cm.viridis(1.2*summ_i['CARINA']['confidence'])
    p_color = cm.viridis(1.2*summ_i['ETT']['confidence'])
    
    
    ax.imshow(img, cmap='gray')
    ax.scatter(*summ_i['CARINA']['pred'], s=7, color=c_color, alpha=0.7, label='pred_carina', vmax = 1)
    ax.scatter(*summ_i['ETT']['pred'], s=7, color=p_color, alpha=0.7, label='pred_ett', vmax = 1)
    ax.scatter(*annot_carina, s=3, c='g', alpha=0.8, label='annot_carina')
    ax.scatter(*annot_probe, s=3, c='g', alpha=0.8, label='annot_ett')
    if not all(np.isnan([err_probe,err_carina])):
        anchored_text = AnchoredText(f'err_carina = {err_carina:.2f}cm\nprobe err = {err_probe:.2f}cm',
                                     frameon=False, loc='lower right')
        ax.add_artist(anchored_text)
    
    ax.set_title(f'CarinaNet on {index}')
    plt.legend(loc='upper right', prop={'size': 6})
    plt.show()
    dataset.save.savefig(fig, dataset.paths.visu_carinaNet(index))
    plt.close()


def _initialize_dict_if_no_carinaNet_output(partial_summary, index):
    for key in ['CARINA', 'ETT']:
        if not key in partial_summary[index]:
            partial_summary[index][key] = {'pred': [np.nan, np.nan], 'confidence': 0}
    return partial_summary

def _summarize(indices, partial_summary):
    for index in indices:
        partial_summary = _initialize_dict_if_no_carinaNet_output(partial_summary, index)
        partial_summary[index]['CARINA']['GT'] = dataset.annot.carina_img(index)
        partial_summary[index]['CARINA']['err'] = dataset.metrics.err1d(partial_summary[index]['CARINA']['pred'], partial_summary[index]['CARINA']["GT"], index)
        partial_summary[index]['ETT']['GT'] = dataset.annot.probe_img(index)
        partial_summary[index]['ETT']['err'] = dataset.metrics.err1d(partial_summary[index]['ETT']['pred'], partial_summary[index]['ETT']["GT"], index)
        if not np.isnan(partial_summary[index]['CARINA']['err']):
            print(f'{index} carina : {partial_summary[index]["CARINA"]["err"]:.2f}cm')
        if not np.isnan(partial_summary[index]['CARINA']['err']):
            print(f'{index} probe : {partial_summary[index]["ETT"]["err"]:.2f}cm')
        
    dataset.summaries.save(partial_summary, 'CarinaNet', 'CarinaNet')
    return partial_summary


def _visualize(indices, summary):
    try :
        for index in indices:
            _visu_CarinaNet(index, summary[index])
    except KeyboardInterrupt :
        print('\n================ Visualization was Keyboard interrupted, exiting\n================')
        return       
                        

def main(indices, plot = True):

    build_retinaNet_csv.build_class_mapping()

    build_retinaNet_csv.build_inference_file(indices)

    with dataset.paths.cd(dataset.paths.retinaNet):
        partial_summary, imgs = inference.main(dataset.paths.traindir)

    summary = _summarize(indices, partial_summary)
    
    if plot:
        _visualize(indices, summary)

    for el in ['CARINA', 'ETT']:
        if all(np.isnan([summ[el]["err"] for summ in summary.values()])):
            print(f'\n    /!\ No annotations were found for {el}-> Displaying no score.')
            
        else:         
            print(f'average {el} error : \
                  {np.nanmean([abs(summ[el]["err"]) for i, summ in summary.items()]):.3f}')
            if len(dataset.test_indices) > 0:
                print(f'average {el} error on test only : \
                      {np.nanmean([abs(summ[el]["err"]) for i, summ in summary.items() if i in dataset.test_indices]):.3f}')

    
