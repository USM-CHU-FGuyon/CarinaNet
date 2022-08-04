# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:10:07 2022

@author: 151985
"""
import torch, argparse, cv2, time, os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from . import retinanet

from .retinanet.dataloader import CSVDataset, collater, Resizer,  \
	UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'


def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def arg_parser(traindir):
    parser = argparse.ArgumentParser(description='Simple script for inference using the transfer learned CarinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', nargs='?', default = 'csv')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)', 
                        default = f'{traindir}class_list.csv')
    parser.add_argument('--model', help='Path to model (.pt) file.', 
                        default = 'model_final.pt')
    
    parser.add_argument('--csv_inference', help='Path to file containing inference files',
                        default = f'{traindir}inference_indices.csv')
    
    return parser.parse_args(None)


def dataset_indices(dataset_val):
    return [os.path.basename(f).split('.')[0] for f in dataset_val.image_names]

def main(traindir):
    parser = arg_parser(traindir)

    dataset_val = CSVDataset(train_file=parser.csv_inference, 
                             class_list=parser.csv_classes, 
                             transform=transforms.Compose([Normalizer(), Resizer()]))

    indices =  dataset_indices(dataset_val)
    
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, 
                                sampler = range(len(dataset_val)))

    
    unnormalize = UnNormalizer()
    
    outputs, imgs = {},{}

    print('\n     Inference...')
    for index, data in zip(indices, dataloader_val):
        outputs[index] = {}
        
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print(f'{index} : Elapsed time: {time.time()-st:.4f}s')
            scores = scores.cpu().numpy()
            classification = classification.cpu().numpy()

            idxs = np.array([np.argmax(scores*(classification==c)) for c in np.unique(classification)]) #Max detection of each class.


            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

        
            for idx in idxs:
                bbox = transformed_anchors[idx, :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idx])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

                pred = [(x1+x2)/(2*data['scale'][0]),
                        (y1+y2)/(2*data['scale'][0])]

                anchor = (transformed_anchors[0].cpu().numpy()/data['scale'][0]).tolist()

                outputs[index][label_name] = {
                                          'confidence': float(scores[idx]),
                                          'anchor':anchor,
                                          'pred' : pred,
                                          'scale':data['scale'][0]
                                          }
            imgs[index] = img
    print('        -> Done')
    return outputs, imgs

    

