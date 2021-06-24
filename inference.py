import cv2 as cv # importing libraries
import numpy as np
import time
import tkinter
from tkinter.filedialog import askopenfilename
import os
from os import walk
import random
import pandas as pd
import shutil

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import argparse


def draw_bbox(image, box, color='green', label=None):
    img = ImageDraw.Draw(image)
    img.rectangle(box, fill=None, outline=color, width=1)
    if label != None:
        img.text((box[0], box[1]), label, fill='red')
    return image

def filter_prediction(prediction, threshold=0.7):
    filtered_prediction = {'labels': list(), 'boxes': list(), 'scores': list()}
    for i in range(len(prediction['labels'])):
        if prediction['scores'][i] >= threshold:
            for s in ['labels', 'boxes', 'scores']:
                filtered_prediction[s].append(prediction[s][i])
    return filtered_prediction

def get_object_detection_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def analyze(path, model, labeldict, device, fold, threshold=0.7, colors=[]):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image_tensor = torchvision.transforms.functional.to_tensor(image)
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])
  
    result = list()
    prediction[0]['boxes'] = np.array(prediction[0]['boxes'].cpu())
    prediction[0]['labels'] = np.array(prediction[0]['labels'].cpu())
    prediction[0]['scores'] = np.array(prediction[0]['scores'].cpu())

    df = pd.DataFrame({'filename': [path]*len(prediction[0]['labels']), 'class': list(map(lambda x: labeldict[x], prediction[0]['labels'])), \
                       'xmin': prediction[0]['boxes'][:, 0], 'ymin': prediction[0]['boxes'][:, 1], 'xmax': prediction[0]['boxes'][:, 2], 'ymax': prediction[0]['boxes'][:, 3], \
                       'score': prediction[0]['scores']})
    df = df[df['score'] >= threshold]

    num_classes = len(labeldict) + 1

    while len(colors) < (num_classes - 1):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        
    color_dict = dict()
    for i in range(1, num_classes):
        color_dict[labeldict[i]] = colors[i-1]
        
    im = Image.fromarray(np.uint8(image))
    for i in range(len(df)):
        draw_bbox(im, df.loc[i][['xmin', 'ymin', 'xmax', 'ymax']], label=df.loc[i]['class'], color=color_dict[df.loc[i]['class']])
    im.save(os.path.join(fold, os.path.split(path)[-1]))
    return df


def inference(model_name, labelmap, files, imsave_path):
    start_time = time.time()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache()

    if os.path.exists(imsave_path):
        shutil.rmtree(imsave_path) # removes output folder if it exists
    os.mkdir(os.path.join(imsave_path)) # makes output folder


    textfile = open(labelmap, 'r') 
    lines = textfile.readlines()
    labels = dict()
    for line in lines:
        label = line.split('\t')
        labels[int(label[1])] = label[0]
    textfile.close()
    num_classes = len(lines) + 1

    model = get_object_detection_model(num_classes)
    model.load_state_dict(torch.load(model_name))
    model.to(device)
    model.eval()
    df = pd.DataFrame()

    for file in files:
        df = pd.concat([df, analyze(file, model, labels, device, imsave_path)])
    print("--- %s seconds ---" % (time.time() - start_time))
    return df

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Script for performing inference.')
    parser.add_argument('input_images', type=str, nargs='+', help='List of images for analyzing.')
    parser.add_argument('model', type=str, help='Path to model.')
    parser.add_argument('labelmap', type=str, help='Path to labelmap.')
    parser.add_argument('-o', type=str, help='Path to output excel file.', default='result.xlsx')
    parser.add_argument('-io', type=str, help='Name of directory for result images. Will be created automatically. Existing folder will be deleted and created again.', default='image_results')
    args = parser.parse_args()

    inference(args.model, args.labelmap, args.input_images, args.io).to_excel(args.o, index=None)
