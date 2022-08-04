import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from Dataset import dataset

def plot(err_dic, test_only=True, annotate_large_err = False):
    """Displays the error separated between the probe and the carina positionning error"""

    if test_only:
        err_dic = dict(err_dic)  # copy
        for key in err_dic:
            err_dic[key] = {i: v for i, v in err_dic[key].items() if i in dataset.test_indices}

    fig, ax = plt.subplots()
    for index, dist in err_dic['dist'].items():
        r = ax.scatter(err_dic['ETT'][index], err_dic['CARINA'][index], c=abs(dist), alpha=0.5, vmin=0, vmax=7.5)
        if annotate_large_err :
            if abs(err_dic['ETT'][index])>2.5 or abs(err_dic['CARINA'][index])>2.5:
              ax.annotate(index, (err_dic['ETT'][index], err_dic['CARINA'][index]))
    ax.set_xlabel('ETT error [cm]')
    ax.set_ylabel('Carina error [cm]')

    mean_p_error = np.mean(np.abs([*err_dic['ETT'].values()]))
    std_p_error = np.std([*err_dic['ETT'].values()])
    mean_c_error = np.mean(np.abs([*err_dic['CARINA'].values()]))
    std_c_error = np.std([*err_dic['CARINA'].values()])
    mean_error = np.mean(np.abs([*err_dic['dist'].values()]))
    std_error = np.std([*err_dic['dist'].values()])
    median_error = np.median(np.abs([*err_dic['dist'].values()]))
    q75, q25 = np.percentile(np.abs([*err_dic['dist'].values()]), [75 ,25])

    ax.annotate(f'Distance error = {mean_error:.2f}cm\n'
                f'Distance median error = {median_error:.2f}cm\n'
                f'ETT mean error = {mean_p_error:.2f}cm\n'
                f'Carina mean error = {mean_c_error:.2f}cm',
                xy=(0.05, 0.75), xycoords='axes fraction', fontsize=8,
                bbox=dict(facecolor='none', edgecolor='k', pad=3))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(r, cax=cax)
    cbar.solids.set(alpha=1)
    cbar.ax.set_ylabel('ETT-Carina distance error')
    ax.set_title(f'Error on {len(err_dic["dist"])} images')
    path = f'{dataset.paths.figures}probe_and_carina_error{"_test_only" * test_only}'

    dataset.save.fig_and_pickle(fig, path, dpi=1000)
    