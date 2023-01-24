import matplotlib.pyplot as plt
import numpy as np

from Dataset import dataset


def visu_cluster_thinning(msk, centers, ett_pos, index):
    fig, ax  = plt.subplots()
    plt.axis('off')
    ax.imshow(msk, vmax = len(centers))
    ax.scatter(*ett_pos, alpha = 0.5, c = 'r')
    for c in centers:
        ax.scatter(*c[::-1], alpha = 1, c = 'r', s = 2)

    plt.savefig(dataset.paths.cluster_thinning_visu(index), bbox_inches="tight",pad_inches=0)
    plt.show()
    plt.close()
    
    
def visu_clusters(clusters, summary, index):
    xmin, ymin, xmax, ymax = summary['xmin'], summary['ymin'], summary['xmax'], summary['ymax']
    im = np.zeros((ymax-ymin,xmax-xmin))-1
    fig, ax = plt.subplots()    
    plt.axis('off')
    rows, cols = zip(*clusters[:,:2])
    im[rows, cols] = clusters[:,-1]
    
    ax.imshow(im)
    ax.scatter(*summary['pred_roi'], c='r')
    plt.savefig(dataset.paths.cluster_visu(index), bbox_inches="tight",pad_inches=0)
    plt.show()
    plt.close()
        