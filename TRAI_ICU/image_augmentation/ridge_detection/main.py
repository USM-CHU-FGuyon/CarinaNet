
import time, multiprocessing

from image_augmentation.ridge_detection import ridge_detection


def _RIDGE_multiprocess(indices):
    n_cpu = multiprocessing.cpu_count()
    print('Starting', n_cpu, 'processes')
    p = multiprocessing.Pool(n_cpu)
    p.map(ridge_detection.run, [[index] for index in indices])


def main(indices):
    print('o Starting Edge Detecton')
    t0 = time.time()
    _RIDGE_multiprocess(indices)
    print(f'   ---> Elapsed :{time.time() - t0} sec')