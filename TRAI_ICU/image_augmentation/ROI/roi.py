"""
Use the prediction of CarinaNet to get a bounding box of the ETT. This is to be used for image enhancement.
"""

import time
from Dataset import dataset


def _summarize(pred, xmin, ymin, xmax, ymax):
    
    pred_roi = round(pred[0]-xmin), round(pred[1]-ymin)
    
    return {
            'xmin':xmin,
            'xmax':xmax,
            'ymin':ymin,
            'ymax':ymax,
            'pred':pred,
            'pred_roi':pred_roi
        }



def main(indices):
    t0 = time.time()
    print('   ROI EXTRACTION')
    carinaNet_summary = dataset.summaries.load('CarinaNet', 'CarinaNet')
    roi_summary = dataset.summaries.load('image_augmentation', 'roi')
    
    for index in indices:
        pred = carinaNet_summary[index]['ETT']['pred']

        xmin = int(pred[0]-150)
        xmax = int(pred[0]+150)
        ymin = 0
        ymax = int(pred[1]+100)

        img = dataset.load.image(index)
        
        Y, X = img.shape

        roi = img[ymin:min(ymax, Y), max(0,xmin):min(xmax, X)]

        dataset.save.ETT_roi(roi, index)
        
        roi_summary[index] = _summarize(pred, xmin, ymin, xmax, ymax )

    dataset.summaries.save(roi_summary, 'image_augmentation', 'roi')
    print(f'       -> Done in {time.time() - t0:.2f}s\n')