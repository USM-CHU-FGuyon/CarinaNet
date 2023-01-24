import time

from Dataset import dataset
from preprocessing import preprocess_images, summarize, dashboard, visualize
from CarinaNet import main as CarinaNet_main
from CarinaNet.CarinaNet import dashboard as CarinaNet_dashboard
from CarinaNet import final_plotting 
from image_augmentation import image_augmentation


def preprocessing(indices):
    t0 = time.time()
    print(f'\no STARTING PREPROCESSING ON {len(indices)} images')

    preprocess_images.main(indices)

    summarize.main()

    dashboard.main()

    visualize.main(indices)
    print(f'PREPROCESSING DONE. Elapsed : {time.time()-t0:.2f}seconds')
    print('========================\n\n')

def inference(indices):
    t0 = time.time()
    print(f'\no STARTING INFERENCE ON {len(indices)} images')
    CarinaNet_main.main(plot=True,
             indices=indices)

    CarinaNet_dashboard.main()
    print(f'INFERENCE DONE. Elapsed : {time.time() - t0:.2f}seconds')
    print('========================\n\n')


def figures_plotting():
    t0 = time.time()
    print(f'\no STARTING PLOTTING')
    final_plotting.main()
    print(f'PLOTTING FIGURES DONE. Elapsed : {time.time() - t0:.2f}seconds')
    print('========================\n\n')

def image_augment(indices):
    t0 = time.time()
    print(f'\no STARTING IMAGE AUGMENTATION ON {len(indices)} IMAGES')
    image_augmentation.main(indices)
    print(f'IMAGE AUGMENTATION DONE. Elapsed : {time.time() - t0:.2f}seconds')
    print('========================\n\n')


def main(indices):
    if not dataset.INFERENCE_MODE:
        raise ValueError('/!\ main.py should only be used with INFERENCE_ONLY=False in Dataset/__init__.py')

    preprocessing(indices)

    inference(indices)

    figures_plotting()

    image_augment(indices)


if __name__ == '__main__':

    indices = [*dataset.indices]

    main(indices)

