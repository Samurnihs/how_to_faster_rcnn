import os
from os import walk
import argparse

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import transforms as T
import bio_rectangular_dataset as bs
#import confusion_matrix as cm
from engine import train_one_epoch, evaluate

import time
import random
import numpy as np
import pandas as pd

import cv2 as cv
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
torch.cuda.empty_cache()

def get_transform(train, tr):
    transforms = []
    if train:
        if '1' in tr:
            transforms.append(T.RandomHorizontalFlip(0.5))
        if '2' in tr:
            transforms.append(T.RandomVerticalFlip(0.5))
        if '3' in tr:
            transforms.append(T.RandomRotation())
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def get_object_detection_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def train(csv, lb_map, tr, num_epochs=10, batch=1):
    # use our dataset and defined transformations
    dataset = bs.bio_data_rectangular('train', csv, lb_map, get_transform(True, tr))
    dataset_test = bs.bio_data_rectangular('test', csv, lb_map, get_transform(False, tr))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fl = open(lb_map, 'r') # we have to know number of classes
    lines = fl.readlines()
    fl.close()
    # we should add one additional class for background
    num_classes = len(lines) + 1

    # get the model using our helper function
    model = get_object_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch + 1) % 5 == 0:
            evaluate(model, data_loader_test, device=device)
    
    return model

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Trains Faster R-CNN ResNet50 model.')
    parser.add_argument('csv', type=str, help='Path to csv file with annotation results.')
    parser.add_argument('lbmap', type=str, help='Path to labelmap txt file.')
    parser.add_argument('-ne', type=int, help='Number of epochs.', default=10)
    parser.add_argument('-bs', type=int, help='Batch size.', default=1)
    parser.add_argument('-tr', type=str, help='List of transformations. 1-Random Horizontal Flip, 1-Random Vertical Flip, 1-Random Rotation.', default='123')
    parser.add_argument('-out', type=str, help='Output model name.', default=os.path.join(os.getcwd(), 'model'))
    args = parser.parse_args()

    model = train(args.csv, args.lbmap, args.tr, args.ne, args.bs)
    #cm.bulid_matrix(model, 'test', args.csv, args.lbmap)
    torch.save(model.state_dict(), args.out)
