import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit 

from Dataset import dataset


def _fit_fun(x, a, b, c): 
    return -a * np.exp(-b * x) + c 


def _confidence_ROC_plot(Ns, errs, confidence_bins, el, test_only, expfit = False):
    fig, ax = plt.subplots()
    
    r = ax.scatter(confidence_bins, errs, c = Ns, cmap = 'Reds', s= 30)
    rr = ax.scatter(confidence_bins, errs, marker = 'o', s = 30)
    rr.set_facecolors('none')
    rr.set_edgecolors('gray')
    ax.set_xlabel('Confidence score')
    ax.set_ylabel('Mean Absolute Error (cm)')
    #ax.set_title(f'Number of images vs Error using confidence on {el} {test_only*"test only"}')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(r, cax=cax)
    cbar.ax.set_ylabel('Number of images')
    
    if expfit:
        try :
            popt, _ = curve_fit(_fit_fun, confidence_bins[1:-1], errs[1:-1])
            ax.plot(np.linspace(0.15,0.9,100), _fit_fun(np.linspace(0.15,0.9,100), *popt), 'r-', label="Fitted Curve")

            ax.annotate(f'Err = {-popt[0]:.1f} * exp(-{popt[1]:.1f} * confidence) {popt[2]:+.1f} ',
                        xy=(0.25, 0.90), xycoords='axes fraction',
                        bbox=dict(facecolor='none', edgecolor='k', pad=3))
        except (ValueError, RuntimeError) as e :
            print('/ ! \ Could not make expfit : Raised error : ', e)
    #ax.plot(np.linspace(0.15,0.9,100), _fit_fun(np.linspace(0.15,0.9,100), -7.1, 4.1, 0.1), 'k-', label="Fitted Curve")
    plt.show()
    dataset.save.fig_and_pickle(fig, f'{dataset.paths.figures}ROC_confidence_{el}_{test_only*"test_only"}.png',
                                bbox_inches='tight')
    
    
    
def _plot_el_roc(carinaNet_summary, confidence_bins, el = '"ETT" or "CARINA"', test_only = True):

    if test_only:
        el_loc_summary = {index: v[el] for index, v in carinaNet_summary.items() if index in dataset.test_indices}
    else :
        el_loc_summary = {index: v[el] for index, v in carinaNet_summary.items()}


    errs, Ns = [], []
    
    def _compute_err(el_loc_summary, confmin, confmax):
        dic = {index: val for index, val in el_loc_summary.items() if (val['confidence']>confmin) and (val['confidence']<confmax)}
        
        errors = [val['err'] for val in dic.values()]
        return np.nanmean(np.abs(errors)), len(errors)
    
    for confmin, confmax in zip(confidence_bins[:-1], confidence_bins[1:]):
        err, N = _compute_err(el_loc_summary, confmin, confmax)
        errs.append(err)
        Ns.append(N)
    _confidence_ROC_plot(Ns, errs, confidence_bins[:-1], el, test_only = test_only)
    
def _model_confience(carinaNet_summary, index):
    return min([carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence']])
    
def _plot_dist_roc(carinaNet_summary, err_dic, confidence_bins, test_only= True):

    errs, Ns = [], []
    
    def _err_size(carinaNet_summary, err_dic, confmin, confmax, test_only = test_only):

        confidence = {index : _model_confience(carinaNet_summary, index) for index in err_dic['dist']}

        dic = {index: val for index, val in err_dic['dist'].items() 
                       if (confidence[index]>confmin ) and (confidence[index] <confmax) }

        if test_only: 
            errors = [val for i, val in dic.items() if i in dataset.test_indices]
        else : 
            errors = [*dic.values()]
        return np.nanmean(np.abs(errors)), len(errors)
    
    for confmin, confmax in zip(confidence_bins[:-1], confidence_bins[1:]):
        err, N = _err_size(carinaNet_summary, err_dic, confmin, confmax)
        errs.append(err),
        Ns.append(N)
    
    _confidence_ROC_plot(Ns, errs, confidence_bins[:-1], 'ETT_carina_dist', expfit = True,
                         test_only = test_only)
    
def plot_roc(carinaNet_summary, err_dic):
    
    confidence_bins = np.linspace(0.15,1,10)
    
    _plot_el_roc(carinaNet_summary, confidence_bins, 'CARINA', test_only = True)
    _plot_el_roc(carinaNet_summary, confidence_bins, 'CARINA', test_only = False)
    _plot_el_roc(carinaNet_summary, confidence_bins, 'ETT', test_only = True)
    _plot_el_roc(carinaNet_summary, confidence_bins, 'ETT', test_only = False)

    _plot_dist_roc(carinaNet_summary, err_dic, confidence_bins, test_only = True)
    _plot_dist_roc(carinaNet_summary, err_dic, confidence_bins, test_only = False)
    
