
from image_augmentation.ridge_detection import ridge_detection
from image_augmentation.ROI import roi
from image_augmentation.clustering import clustering
from image_augmentation.clustering import cluster_selection
from image_augmentation.image_enhancement import image_enhancement


def main(indices):

    roi.main(indices)

    ridge_detection.run(indices)

    clustering.main(indices)

    cluster_selection.main(indices)
    
    image_enhancement.main(indices)
    
