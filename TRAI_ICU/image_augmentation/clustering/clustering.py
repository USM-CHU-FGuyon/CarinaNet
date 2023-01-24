import time

import sklearn.cluster
import numpy as np

from Dataset import dataset


def _DBSCAN_clustering(edg_binary, index):
    """ANIsotropic DBSCAN clustering : some documentation would be nice here :)
    returns an array with """

    X = np.array(np.nonzero(edg_binary)).transpose()
    y_anisotropy = 1

    X[:, 1] = X[:, 1]*y_anisotropy
    eps, min_samples = 5, 10
    db = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    X[:, 1] = X[:, 1] * 1 / y_anisotropy
    labelled_points = np.column_stack((X, db.labels_))

    return labelled_points

def _spectral_clusetring(edg_binary, index):
    'actually write it'

def _clustering(edg_binary, index, METHOD):
    if METHOD == 'DBSCAN':
        return _DBSCAN_clustering(edg_binary, index)
    if METHOD == 'SPECTRAL':
        return _spectral_clusetring(edg_binary, index)


def _run_clustering(indices, METHOD):
    for index in indices:
        edge_map = dataset.load.image_augment_edges(index)
        edges = edge_map > np.quantile(edge_map, 0.90)  # foolishly convert to a binary
        
        labelled_points = _clustering(edges, index, METHOD)
        
        dataset.save.edges_binary(edges, index)
        dataset.save.image_augment_clusters(labelled_points, index)
        


def main(indices, METHOD = 'DBSCAN'):
    print(f'   CLUSTERING USING {METHOD}')
    t0 = time.time()

    _run_clustering(indices, METHOD)

    print(f'      -> Done in :{time.time() - t0:.2f}s\n')
