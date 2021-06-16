import os
import random
import argparse
import shutil


def dist_fold(img_path_list, fract=0.8): # divides images into train and test set. Defaul part of test images is 80%
    for img_path in img_path_list:
        if os.path.exists('train'):
            shutil.rmtree('train') # removes train folder if it exists
        if os.path.exists('test'):
            shutil.rmtree('test') # removes train folder if it exists
        os.mkdir(os.path.join(os.getcwd(), 'train')) # makes train folder
        os.mkdir(os.path.join(os.getcwd(), 'test')) # makes test folder
        files = list(map(lambda x: os.path.join(img_path, x), os.listdir(img_path))) # obtains list of all images
        files = list(filter(lambda x: not(x.endswith('.xml')), files)) # we need take only images
        random.shuffle(files) # shuffling files randonmly
        if fract <= 1: # thus we can indicate fraction or number of images in training set
            thr = int(len(files) * fract)
        else:
            thr = fract # fract > 1 will be parsed as absolute number in test set
        
        for i in range(len(files)):
            if i < thr:
                shutil.copyfile(files[i], os.path.join('train', os.path.split(files[i])[1]))
            else:
                shutil.copyfile(files[i], os.path.join('test', os.path.split(files[i])[1]))


if __name__ =='__main__':
    parser = argparse.ArgumentParser( #parsing arguments
    description='Distributes all images into train and test sets randomly.')
    parser.add_argument('input_folder', type=str, nargs='+', help='Path to folder(s) with images.')
    parser.add_argument('-fr', type=float, help='Fraction of images in test set.', default=0.8)
    parser.add_argument('-abs', type=int, help='Absolute number of images in test set. Will be used first if set.', default=0)
    args = parser.parse_args()
    
    
    dist_fold(args.input_folder, max(args.fr, args.abs))
