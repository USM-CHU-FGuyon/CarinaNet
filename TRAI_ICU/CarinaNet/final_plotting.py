import numpy as np
import pandas as pd

from .dashboard import dashboard
from Dataset import dataset


def _compute_spacing(both_annot, carinaNet_summary):
    to_cm = dataset.metrics.to_cm
    indices= dataset.indices
    err_dic = {'dist': {}, 'ETT': {}, 'CARINA': {}}
    spacing = {}
    preds = {}
    for index in both_annot:
        if index in carinaNet_summary : #probe_loc_summary and index in carina_loc_summary:
            

            preds[index] = {'pred_carina': carinaNet_summary[index]['CARINA']['pred'][1],
                            'pred_ett': carinaNet_summary[index]['ETT']['pred'][1],
                            'confidence': min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence']),
                            'gt_carina':carinaNet_summary[index]['CARINA']['GT'][1],
                            'gt_ett': carinaNet_summary[index]['ETT']['GT'][1],
                            'to_cm': to_cm(index),
                            'img_pth': indices[index]['path'],
                            'is_test': index in dataset.test_indices
                            }
            detected_spacing = preds[index]['pred_carina'] - preds[index]['pred_ett']
            GT_spacing = preds[index]['gt_carina'] - preds[index]['gt_ett']

            spacing[index] = {'pred': detected_spacing * to_cm(index),
                              'GT': GT_spacing * to_cm(index)}

            # Compute the error
            err_dic['dist'][index] = (GT_spacing - detected_spacing) * to_cm(index)
            err_dic['ETT'][index] = to_cm(index) * (carinaNet_summary[index]['ETT']["pred"][1] - carinaNet_summary[index]['ETT']["GT"][1])
            err_dic['CARINA'][index] = to_cm(index) * (
                        carinaNet_summary[index]['CARINA']["pred"][1] - carinaNet_summary[index]['CARINA']["GT"][1])
            print(f'{index} : {err_dic["dist"][index]:.2f}cm, '
                  f'probe_error : {err_dic["ETT"][index]:.2f}cm, '
                  f'carina_error : {err_dic["CARINA"][index]:.2f}cm')
    
    pd.DataFrame.from_dict(preds).transpose().to_csv(dataset.paths.preds)
    return err_dic, spacing

def high_confidence_error(carinaNet_summary, err_dic, conf = 0.3):

    dist_err_conf = {index : val for index, val in err_dic['dist'].items() if min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence']) > conf}
    carina_err_conf = {index : val for index, val in err_dic['CARINA'].items() if min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence']) > conf}
    ETT_err_conf = {index : val for index, val in err_dic['ETT'].items() if min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence']) > conf}

    meanerr_dist = np.mean(np.abs(list(dist_err_conf.values())))
    meanerr_ett = np.mean(np.abs(list(ETT_err_conf.values())))
    meanerr_carina = np.mean(np.abs(list(carina_err_conf.values())))
    
    stderr_dist = np.std(list(dist_err_conf.values()))
    stderr_ett = np.std(list(carina_err_conf.values()))
    stderr_carina = np.std(list(ETT_err_conf.values()))

    print(f"Error on confidence > {conf:.2f} : {len(dist_err_conf)} images\n"
          f" dist :    {meanerr_dist:.2f}cm std : {stderr_dist:.2f}\n"
          f" ETT    : {meanerr_carina:.2f}cm std : {stderr_carina:.2f}\n"
          f" Carina : {meanerr_ett:.2f}cm std {stderr_ett:.2f}\n\n")

def median_confidence(carinaNet_summary):
    conf_ett = [carinaNet_summary[index]['ETT']['confidence'] for index in carinaNet_summary]
    conf_carina = [carinaNet_summary[index]['CARINA']['confidence'] for index in carinaNet_summary]
    print(f'median ETT confidence : {np.median(conf_ett)}')
    print(f'median Carina confidence : {np.median(conf_carina)}')


def main():
    if dataset.INFERENCE_MODE :
        print('INFERENCE MODE -> No plotting')
        return

    both_annot = dataset.annot.annoted_probe_AND_carina()

    print(f'Found {len(both_annot)} images where probe and carina are annotated.')
    if len(both_annot)> 0:
        carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')

        err_dic, spacing = _compute_spacing(both_annot, carinaNet_summary)

        median_confidence(carinaNet_summary)

        dashboard.main(carinaNet_summary, err_dic, spacing)
        for conf in np.linspace(0,0.9, 10):
            high_confidence_error(carinaNet_summary, err_dic, conf = conf)

