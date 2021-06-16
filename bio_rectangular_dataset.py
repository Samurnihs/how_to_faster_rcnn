import os
from os import walk

import torch
import torch.utils.data

import numpy as np
import pandas as pd

import cv2 as cv
from PIL import Image

class bio_data_rectangular(torch.utils.data.Dataset):
    def __init__(self, root, annotations, labels, transforms=None):
        # root directory
        self.root = root 
        # list of transforms
        self.transforms = transforms 
        # lists for images and annotations 
        self.imgs = list(sorted(os.listdir(self.root)))
        self.info = pd.read_csv(annotations)
        # getting list of labels
        self.labelnames = list()
        self.labels = dict()
        file = open(labels, 'r')
        lines = file.readlines()
        for line in lines:
            label = line.split('\t')
            self.labelnames.append(label[0])
            self.labels[label[0]] = int(label[1])
        file.close()
        
    # функция, позволяющая получить изображение из набора и информацию о нём
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.fromarray(np.uint8(np.array(cv.imread(img_path))))
        # получение всей информации о выбранном изображении из аннотаций
        img_info = self.info.loc[self.info['filename'] == self.imgs[idx]].reset_index(drop=True)
        # число объектов
        num_objs = len(img_info)
        # классы объектов
        labels_prev = img_info.loc[:, 'class']
        # координаты прямоугольников, содержащих объекты
        boxes_prev = img_info.loc[:, 'xmin':'ymax']
        boxes = []
        labels = []
        # перевод текстовых значений классов в числовые
        for i in range(num_objs):
            labels.append(self.labels[labels_prev.loc[i]])
            boxes.append(np.array(boxes_prev.loc[i]))
        # приведение всей информации к виду тензоров torch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        # вычисление площадей прямоугольников с объектами       
        area = torch.as_tensor((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        # словарь с информацией об изображении
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # применение трансформаций изображений
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        # возвращение изображения и информации о нём
        return img, target
    # функция, возвращающая длину набора данных
    def __len__(self):
        return len(self.imgs)
