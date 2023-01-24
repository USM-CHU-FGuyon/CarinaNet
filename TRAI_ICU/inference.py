import sys, os, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Dataset import dataset
from preprocessing import preprocess_images, summarize, dashboard, visualize
from CarinaNet import main as CarinaNet_main
from CarinaNet.CarinaNet import dashboard as CarinaNet_dashboard
from image_augmentation import image_augmentation
from CarinaNet import final_plotting


def preprocessing(indices):
    print('o STARTING PREPROCESSING')

    preprocess_images.main(indices)

    summarize.main()

    dashboard.main()

    visualize.main(indices)
    print('========================\n\n')


def inference(indices):
    print('o STARTING INFERENCE')
    CarinaNet_main.main(plot=True,
                        indices=indices)

    CarinaNet_dashboard.main()
    print('========================\n\n')


def main(dirname):
    if not dataset.INFERENCE_MODE:
        raise ValueError('/!\ inference.py should only be used with the INFERENCE_ONLY mode in Dataset/__init__.py')

    dataset.init_inference(dirname)

    indices = dataset.indices

    preprocessing(indices)

    inference(indices)

    final_plotting.main()

    image_augmentation.main(indices)

    return [dataset.paths.augmented_image(i) for i in indices]


if __name__ == '__main__':
    t0 = time.time()
    main('../data/example/images/')
    print(f'TOTAL INFERENCE TIME : {time.time() - t0:.1f}s')