import matplotlib.pyplot as plt
import numpy as np
import time
from Dataset import dataset


def _uncertainty_quantification(index, carinaNet_summary):
    """Uses a correlation between the confidence and the average error"""
    min_confidence = min(carinaNet_summary[index]['ETT']['confidence'], carinaNet_summary[index]['CARINA']['confidence'])
    return 4.3*np.exp(-3.3*min_confidence)+0.1 #average error in cm

def _alter_image(img, points, ETT, carina, uncertainty, index):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img, cmap = 'gray')
    ax.scatter(*zip(*points[:,::-1]), alpha = 1, c = 'r', s = 3, label = 'ETT')
    ax.scatter(*ETT, c = 'purple', s = 5, alpha = 1, label = 'ETT tip')
    ax.scatter(*carina, c = 'green', s = 5, alpha = 1, label = 'Carina')
    dist = dataset.metrics.to_cm(index) * (carina[1]-ETT[1])
    ax.annotate(f'ETT-carina distance = {dist:.1f}cm $\pm$ {uncertainty:.1f}cm',
                xy=(0.25, 0.05), xycoords='axes fraction', fontsize=8,
                bbox=dict(facecolor='none', edgecolor='k', pad=3), color = 'k')
    plt.legend()
    #plt.show()
    plt.savefig( dataset.paths.augmented_image(index), bbox_inches="tight", pad_inches=0)

def main(indices):
    t0 = time.time()
    print('   CREATING OUTPUT')
    carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')
    for index in indices:
        img = dataset.load.image(index)
        points = dataset.load.ETT_detection(index)
        ETT = carinaNet_summary[index]['ETT']['pred']
        carina = carinaNet_summary[index]['CARINA']['pred']
        uncertainty = _uncertainty_quantification(index, carinaNet_summary)
        _alter_image(img, points, ETT, carina, uncertainty, index)

    print(f'      -> Done in :{time.time() - t0:.2f}s\n')
