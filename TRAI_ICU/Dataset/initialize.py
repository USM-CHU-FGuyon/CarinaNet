# -*- coding: utf-8 -*-
import os


def _makedir(path, pathHandler):
    if '.' in path.split('/')[-1]:
        path = '/'.join(path.split('/')[:-1])

    try : 
        os.makedirs(path)
        print(f'   -> Created {pathHandler.printable(path)}')
    except FileExistsError:
        return

def mkdir_outputs(pathHandler):
    """
    Creates a directory ```../outputs/name/``` with all the needed subdirectories for saving the results.

    Parameters
    ----------
    pathHandler : PathHandler object of the Dataset.

    Returns
    -------
    None.
    """

    print('o INITIALIZING THE OUTPUT DIRECTORY... ')


    _makedir(pathHandler.metadata_path, pathHandler)
    _makedir(pathHandler.figures, pathHandler)
    for phase in ['data', 'CarinaNet']:
        _makedir(pathHandler.dashboard_dir(phase), pathHandler)
    
    for f in [pathHandler.image('0'),
              pathHandler.compressed_image('0'),
              pathHandler.annot_visu('0'),
              pathHandler.visu_carinaNet('0'),
              pathHandler.traindir,
              pathHandler.ETT_roi('0'),
              pathHandler.image_augment_edges('0'),
              pathHandler.image_augment_ridges('0'),
              pathHandler.image_augment_clusters('0'),
              pathHandler.augmented_image('0'),
              pathHandler.ETT_detection('0'),
              pathHandler.image_augment_closest_cluster('0'),
              pathHandler.cluster_thinning_visu('0'),
              pathHandler.edges_binary('0'),
              pathHandler.cluster_visu('0')]:
        _makedir(f, pathHandler)

    print('      -> Done.')