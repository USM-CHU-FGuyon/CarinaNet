import time
import numpy as np

from Dataset import dataset
from . import cluster_utils


def _closest_cluster(clusters, pos):

    distances = np.sum(abs(clusters[:,:2]-pos[::-1]), axis = 1)

    closest_point = clusters[np.argmin(distances)]
    closest_cluster = clusters[clusters[:,2]==closest_point[2]]
    return closest_cluster



def _cluster_thinning(cluster, ett_pos, summary, index):
    """Select the ett in the cluster. method is slow, and should be easy to speedup."""

    xmin, ymin, xmax, ymax = summary['xmin'], summary['ymin'], summary['xmax'], summary['ymax']
    
    cluster = cluster[:, :2]
    msk = np.zeros((ymax-ymin,xmax-xmin))
    rows, cols = zip(*cluster)
    msk[rows, cols] = 25
    
    dist = np.sum(np.abs(cluster-ett_pos[::-1]), axis = 1)
    
    closest_point = cluster[np.argmin(dist)]
    msk[closest_point[0]:] = 0 #remove points under closest detection
    msk[closest_point[0], closest_point[1]] = 2
    
    
    centers = [closest_point]
    radius = 10
    while centers[-1][0]>10:
        region_around_center = cluster[np.sum(np.abs(cluster-centers[-1]), axis =1)<radius]
        if len(region_around_center)==0:
            break
        rows, cols = zip(*region_around_center)
        msk[rows, cols] = len(centers)+1
        center = np.mean(region_around_center, axis = 0)

        cluster = cluster[cluster[:, 0]<center[0]]
        centers.append(center)
    
    cluster_utils.visu_cluster_thinning(msk, centers, ett_pos, index)

    return np.array(centers)+np.array([ymin,xmin])
    
def main(indices):
    t0 = time.time()
    print(f'   CLUSTER THINNING')
    roi_summary = dataset.summaries.load('image_augmentation', 'roi', strict = True)
    
    for index in indices:
        clusters = dataset.load.image_augment_clusters(index)
        
        ett_pos = roi_summary[index]['pred_roi'] #predicted position of the ett tip in the roi
        
        cluster_utils.visu_clusters(clusters, roi_summary[index], index)
        
        cluster = _closest_cluster(clusters, ett_pos)
        
        dataset.save.closest_cluster(cluster, index)
        
        ett_points = _cluster_thinning(cluster, ett_pos, roi_summary[index], index)

        dataset.save.ETT_detection(ett_points, index)

    print(f'      -> Done in :{time.time() - t0:.2f}s\n')


