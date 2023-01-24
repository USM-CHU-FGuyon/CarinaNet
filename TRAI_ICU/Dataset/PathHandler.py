from . import json_utils
from contextlib import contextmanager
import os
from pathlib import Path
"""
This class lists all the filenames of useful files such as 
    * visualization plots
    * summaries
    * image paths at each step

"""

        
class PathHandler(object):

    def __init__(self, datasetname, path_to_img, xls_annot_path):
        self.base = os.path.dirname(__file__)
        self._print_base = Path(self.base).parent #base path from which paths are printed
        self.metadata_path = f'{self.base}/{datasetname}/'
        
        self.outputdir = f'{self.base}/../../outputs/{datasetname}/'
        self.preds = f'{self.outputdir}CarinaNet/preds.csv'
        self.classifications = f'{self.outputdir}CarinaNet/classifications.csv'
        self.indices = f'{self.outputdir}indices.json'
        self.annotations = f'{self.metadata_path}annotations.json'
        self.traindir = f'{self.outputdir}CarinaNet/train/'
        #self.hist_traindir = f'{self.base}/../CarinaNet/CarinaNet/train_hist_0604/'
        self.hist_traindir = f'{self.base}/../CarinaNet/CarinaNet/train_hist/'
        self.figures = f'{self.outputdir}figures/'
        self.retinaNet = f'{self.base}/../CarinaNet/CarinaNet/pytorch_retinanet/'
        self.xls_annot_path = xls_annot_path
        self.db_path = path_to_img+'/'
        
    def printable(self, pth):
        return os.path.relpath(pth,self._print_base)

    def load_jsondata(self):
        return json_utils.loadjson(f'{self.metadata_path}path_to_database.json')
        
    #GENERAL

    def summary(self, phase, step):
        return f'{self.outputdir + phase}/{step}_summary.json'

    def dashboard_dir(self, phase):
        return f'{self.outputdir + phase}/dashboard/'

    #0 PREPROCESSING

    def db_image(self, index):
        return self.db_path + index + '.jpg'

    def image(self, index):
        return f'{self.outputdir}data/images/{index}.png'

    def compressed_image(self, index):
        return f'{self.outputdir}data/compressed_images/{index}.jpg'

    def annot_visu(self, index):
        return f'{self.outputdir}data/visu/{index}.png'


    # CarinaNet

    def visu_carinaNet(self, index):
        return f'{self.outputdir}CarinaNet/visu/{index}.jpg'

    #Image augmentation

    def cluster_visu(self, index):
        return f'{self.outputdir}image_augmentation/visu_clusters/{index}.png'

    def cluster_thinning_visu(self, index):
        return f'{self.outputdir}image_augmentation/visu_cluster_thinning/{index}.png'

    def ETT_roi(self, index):
        return f'{self.outputdir}image_augmentation/1_ROI/{index}.png'

    def image_augment_ridges(self, index):
        return f'{self.outputdir}image_augmentation/2_ridges/{index}.png'

    def image_augment_edges(self, index):
        return f'{self.outputdir}image_augmentation/3_edges/{index}.png'

    def edges_binary(self, index):
        return f'{self.outputdir}image_augmentation/4_edges_binary/{index}.png'

    def image_augment_clusters(self, index):
        return f'{self.outputdir}image_augmentation/5_clusters/{index}.npy'

    def image_augment_closest_cluster(self, index):
        return f'{self.outputdir}image_augmentation/6_closest_cluster/{index}.npy'

    def ETT_detection(self, index):
        return f'{self.outputdir}image_augmentation/7_ETT_points/{index}.npy'

    def augmented_image(self, index):
        return f'{self.outputdir}image_augmentation/8_augmented_image/{index}.png'

    @contextmanager
    def cd(self, newdir):
        prevdir = os.getcwd()
        os.chdir(os.path.expanduser(newdir))
        try:
            yield
        finally:
            os.chdir(prevdir)
    