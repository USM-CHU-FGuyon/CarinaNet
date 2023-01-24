import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from numpy.random import default_rng
import numpy as np

from Dataset import dataset

def _confusion_matrix(spacing, reader_func, name):

    GT_spacing = {index: spacing[index]['GT']>2 for index in spacing if index in dataset.test_indices}
    pred = {index:  reader_func(index, spacing)  for index in spacing if index in dataset.test_indices}


    fig, ax = plt.subplots()
    cmd = ConfusionMatrixDisplay.from_predictions([*GT_spacing.values()], [*pred.values()],
                                            cmap = 'Blues', 
                                            ax = ax,
                                            display_labels = ['Too Low', 'Good'])
    cmd.ax_.set(xlabel = f'{name}', ylabel = 'Effective ETT-carina distance')

    plt.show()
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}confusion_matrix_{name}.png')
    return pred, GT_spacing


def _reader_classif(index, spacing):
    return dataset.annot.ETT_position(index) in [0,1,2]

def _AI_classif(index, spacing, min_dist = 2):
    carinanet_s = dataset.summaries.load('CarinaNet','CarinaNet')
    u = dataset.metrics.uncertainty(min(carinanet_s[index]['CARINA']['confidence'],carinanet_s[index]['ETT']['confidence']))
    return bool((spacing[index]['pred'] - u) > min_dist)

def _AI_and_reader(index, spacing):
    return _AI_classif(index, spacing) and _reader_classif(index, spacing)

def _save_classifications(spacing, GT, reader_pred, AI_pred):

    classification = {
        index :{ 'GT':GT[index],
                 'reader': reader_pred[index],
                 'AI': AI_pred[index]}
        for index in GT.keys()
    }
    dataset.summaries.save(classification, 'CarinaNet', 'classification')
    
    (pd.DataFrame([GT, reader_pred, AI_pred], index=['GT', 'reader', 'AI'])
     .transpose()
     .to_csv(dataset.paths.classifications))
    
    
    
    
def se_sp(pred, gt):

    tn = np.sum(gt[pred])
    fp = np.sum(gt[~pred])
    tp = np.sum(1-gt[~pred])
    fn = np.sum(1-gt[pred])
    
    se = tp/(tp+fn)
    sp = tn/(tn+fp)
    return se, sp
    

def _bootstrap(pred, gt, n_bootstrap = 1000, size_bootstrap = 200, title =''):
    print(f'\n{title}')
    assert [*gt.keys()] == [*pred.keys()]
    
    gt = np.array([*gt.values()])
    pred = np.array([*pred.values()])
    se, sp = se_sp(pred, gt)
    print(se,sp)
    ses, sps = [],[]
    for k in range(n_bootstrap):
        rng = default_rng(k)
        bs_pred = rng.choice(pred, size_bootstrap)
        rng = default_rng(k)
        bs_gt = rng.choice(gt, size_bootstrap)
        se_bs, sp_bs = se_sp(bs_pred, bs_gt)
        ses.append(se_bs)
        sps.append(sp_bs)
    IC_se = [np.quantile(ses, 0.025),np.quantile(ses, 0.975)]
    IC_sp = [np.quantile(sps, 0.025),np.quantile(sps, 0.975)]
    
    print(f'Sensitivity : {se}, IC : {IC_se}')
    print(f'Specificity : {sp}, IC : {IC_sp}')
    return se, sp, IC_se, IC_sp

def plot(spacing):

    reader_pred, GT = _confusion_matrix(spacing, _reader_classif, name = 'Clinician')
    AI_pred, _ = _confusion_matrix(spacing, _AI_classif, name = 'CarinaNet model')
    AI_and_reader_pred, _ = _confusion_matrix(spacing, _AI_and_reader, name = 'CarinaNet model + Clinician')

    _save_classifications(spacing, GT, reader_pred, AI_pred)

    _bootstrap(reader_pred, GT, title = 'Clinician')
    _bootstrap(AI_pred, GT, title = 'AI')
    _bootstrap(AI_and_reader_pred, GT, title = 'AI + Clinician')
    
    
    
    
    