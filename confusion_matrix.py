import os
from os import walk
import random
import pandas as pd
import time
import io
import argparse
import cv2 as cv
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt


def files_inside(path):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def get_info(name, data, label_file):
    img_info = data.loc[data['filename'] == name].reset_index(drop=True)
    num_objs = len(img_info)
    labels_prev = img_info.loc[:, 'class']
    boxes_prev = img_info.loc[:, 'xmin':'ymax']
    boxes = []
    labels = []
    
    file = open(label_file, 'r')
    lines = file.readlines()
    labelnames = dict()
    for line in lines:
        label = line.split('\t')
        labelnames[label[0]] = int(label[1])
    file.close()
    
    for i in range(num_objs):
        labels.append(labelnames[labels_prev.loc[i]])
        boxes.append(np.array(boxes_prev.loc[i]))
    return {'name': name, 'labels': labels, 'boxes': boxes}

def inference(path, model, device):
    
    image = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
    
    fin_pred = {'labels': np.array(prediction[0]['labels'].cpu()).astype('int32'),
                'boxes': np.array(prediction[0]['boxes'].cpu()).astype('int32'),
               'scores': np.array(prediction[0]['scores'].cpu()).astype('float32')}
 
    return fin_pred

def filter_prediction(prediction, threshold=0.7):
    filtered_prediction = {'labels': list(), 'boxes': list(), 'scores': list()}
    used = list()
    for i in range(len(prediction['labels'])):
        if prediction['scores'][i] >= threshold:
            for s in ['labels', 'boxes', 'scores']:
                filtered_prediction[s].append(prediction[s][i])
            used.append(False)
    filtered_prediction['used'] = used
    return filtered_prediction

def IoU(ground_truth_box, predicted_box):
    g_xmin, g_ymin, g_xmax, g_ymax = ground_truth_box[0], ground_truth_box[1], ground_truth_box[2], ground_truth_box[3]
    p_xmin, p_ymin, p_xmax, p_ymax = predicted_box[0], predicted_box[1], predicted_box[2], predicted_box[3]
    
    xa = max(g_xmin, p_xmin)
    ya = max(g_ymin, p_ymin)
    xb = min(g_xmax, p_xmax)
    yb = min(g_ymax, p_ymax)

    intersection = max(0, xb - xa) * max(0, yb - ya)

    boxgtArea = (g_xmax - g_xmin) * (g_ymax - g_ymin)
    boxprArea = (p_xmax - p_xmin) * (p_ymax - p_ymin)
    
    if (boxgtArea + boxprArea - intersection) != 0:
        return intersection / float(boxgtArea + boxprArea - intersection)


def get_object_detection_model(num_classes, device):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    return model


def confusion(prediction, annotation, num_classes, iou_threshold=0.5):
    confuse = np.zeros((num_classes, num_classes)).astype('int32')
    for i in range(len(annotation['boxes'])):
        iou_max = 0
        pair_label = num_classes
        index = -1
        for j in range(len(prediction['boxes'])):
            iou_unit = IoU(annotation['boxes'][i], prediction['boxes'][j])
            if (iou_unit >= 0.5):
                if (iou_unit > iou_max):
                    if prediction['used'][j] == False:
                        iou_max = iou_unit
                        pair_label = prediction['labels'][j]
                        index = j
#        print(pair_label)
        confuse[annotation['labels'][i] - 1][pair_label - 1] += 1
#        print(annotation['labels'][i], pair_label)
        if index >= 0:
            prediction['used'][index] = True
    
    for i in range(len(prediction['boxes'])):
        if prediction['used'][i] == False:
            confuse[num_classes - 1][prediction['labels'][i] - 1] += 1
#    print(prediction)
    return(confuse)

#cm.bulid_matrix(model, 'test', args.csv, args.lbmap)
def bulid_matrix(model_p, image_data, csv_name, lbmap): #num_classes):
    files = files_inside(image_data)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    file = open(lbmap, 'r')
    lines = file.readlines()
    num_classes = len(lines)+1


    model = get_object_detection_model(num_classes, device)
    model.load_state_dict(torch.load(model_p))
    model.eval()


    labs = dict()
    clss = list()
    for i in range(len(lines)):
        info = lines[i].split()
        labs[int(info[1])] = info[0]
        clss.append(info[0])

    
    confusion_matrix = np.zeros((num_classes, num_classes)).astype('int32')
    for file in files:
        ground_truth = get_info(file, pd.read_csv(csv_name), lbmap)
        prediction = filter_prediction(inference(os.path.join(image_data, file), model, device))
        confusion_matrix += confusion(prediction, ground_truth, num_classes)
    print('Rows - Actual class, Columns - Predicted class')
    #print(confusion_matrix)

    for i in range(len(confusion_matrix) - 1):
        print('{}. Precision: {}, Recal: {}'.format(labs[(i+1)], (confusion_matrix[i][i]/np.sum(confusion_matrix, axis=0)[i]), (confusion_matrix[i][i]/np.sum(confusion_matrix, axis=1)[i])))
    

    classes = clss + ['nothing']

    plt.xticks(ticks=np.arange(len(classes)),labels=list(map( lambda x: 'Predicted '+ x, classes)), rotation=90)
    plt.yticks(ticks=np.arange(len(classes)),labels=list(map( lambda x: 'Actual '+ x, classes)))

    for i in range(len(classes)):
      for j in range(len(classes)):
        text = plt.text(j, i, confusion_matrix[i, j],
                       ha="center", va="center", color="k")

    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.imshow(confusion_matrix, cmap='BuGn',interpolation="nearest")
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Builds confusion matrix.')
    parser.add_argument('model', type=str, help='Path to deep-learning model.')
    parser.add_argument('lm', type=str, help='Path to labelmap.')
    parser.add_argument('csv', type=str, help='Path to csv file with ground thruth information about images.')
    parser.add_argument('f', type=str, help='Path to images.')
    args = parser.parse_args()


    bulid_matrix(args.model, args.f, args.csv, args.lm)


