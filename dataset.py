"""
  FileName     [ dataset.py ]
  PackageName  [ HW3 ]
  Synopsis     [ Dataset of the HW3: CelebA, USPS, SVHN, MNISTM. ]

  CelebA: 64 * 64 * 3

  USPS: 28 * 28 * 1 -> 28 * 28 * 3
  SVHN: 28 * 28 * 3
  MNISTM: 28 * 28 * 3
"""

import csv
import os
import random
import time

# import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

import utils


class CelebA(Dataset):
    def __init__(self, root, feature, transform=None):
        """ 
          Save the imageNames and the labelNames and read in future.
        """
        self.datas = []
        self.root  = root
        self.feature   = feature
        self.transform = transform

        image_folder = os.path.join(root, "train")
        anno_file    = os.path.join(root, "train.csv")
        dataFrame = pd.read_csv(anno_file)
        
        for _, row in dataFrame.iterrows():
            img_name, keyFeature = row["image_name"], row[feature]
            img_name = os.path.join(image_folder, img_name)
            
            self.datas.append((img_name, keyFeature))
        
        self.len = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, feature = self.datas[index]
        
        img = Image.open(img_name)
        
        if feature == 0:
            feature = torch.Tensor([1, 0])
        elif feature == 1:
            feature = torch.Tensor([0, 1])
        # feature = torch.Tensor(feature)
        
        if self.transform: 
            img = self.transform(img)
        
        return img, feature

class NumberClassify(Dataset):
    def __init__(self, root, feature, train, black=False, transform=None):
        """ 
          Save the imageNames and the labelNames and read in future.
        """
        self.datas = []
        self.root  = root
        self.black = black
        self.feature   = feature
        self.transform = transform

        if train:
            image_folder = os.path.join(root, feature, "train")
            anno_file = os.path.join(root, feature, "train.csv")
        else:
            image_folder = os.path.join(root, feature, "test")
            anno_file = os.path.join(root, feature, "test.csv")

        dataFrame = pd.read_csv(anno_file)
        
        for _, row in dataFrame.iterrows():
            img_name, label = row["image_name"], row["label"]
            img_name = os.path.join(image_folder, img_name)
            
            self.datas.append((img_name, label))
        
        self.len = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name, label = self.datas[index]
        
        img = Image.open(img_name)
        label = torch.Tensor([label])
        
        if self.black:
            img = img.convert("RGB")

        if self.transform: 
            img = self.transform(img)

        return img, label, img_name

class NumberPredict(Dataset):
    def __init__(self, img_folder, black=False, transform=None):
        """ Handling read imgs only """
        self.datas = []
        self.img_folder = img_folder
        self.black      = black
        self.transform  = transform
        
        self.datas = [os.path.join(img_folder, img_name) for img_name in os.listdir(img_folder)]
        self.len   = len(self.datas)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_name = self.datas[index]
        img      = Image.open(img_name)
        
        if self.black:
            img = img.convert("RGB")

        if self.transform: 
            img = self.transform(img)

        return img, img_name

def celebA_unittest():
    features = utils.faceFeatures
    
    print(features[0])
    dataset = CelebA("./hw3_data/face", features[0], transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    dataiter = iter(dataloader)
    img, feature, img_name = next(dataiter)
    
    print(img_name)
    print(feature)
    print(img.shape)
    plt.imshow(img[0].permute(1, 2, 0))
    plt.waitforbuttonpress()
    plt.close()
    
    return

def number_unittest():
    for feature, black, train in [("usps", True, True), ("svhn", False, True), ("mnistm", False, True), ("usps", True, False), ("svhn", False, False), ("mnistm", False, False)]:
        print("{}, Black: {}, Train: {}".format(feature, black, train))

        dataset = NumberClassify("./hw3_data/digits", feature, train=train, black=black, transform=transforms.Compose([
            transforms.ToTensor()
        ]))

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        dataiter = iter(dataloader)
        img, label, img_name = next(dataiter)
        print(img_name)
        print(label)
        print(img.shape)
        plt.imshow(img[0].permute(1, 2, 0))
        plt.waitforbuttonpress()
        plt.close()

    return

def usps_unittest():
    print("USPS, Black: True, Train: True")

    dataset = NumberClassify("./hw3_data/digits", "usps", train=True, black=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    dataiter = iter(dataloader)
    img, label, img_name = next(dataiter)
    print(img_name)
    print(label)
    print(img.shape)
    
    return

def main():
    # celebA_unittest()
    # print("celebA_unittest Passed!")

    # number_unittest()
    # print("number_unittest Passed!")

    usps_unittest()
    print("usps_unittest Passed!")

if __name__ == "__main__":
    main()
