"""
  FileName     [ predict.py ]
  PackageName  [ HW3 ]
  Synopsis     [ Domain Adaptive prediction from DANN(Domain Adaptive Neural Network). ]

  Dataset:
    USPS: 28 * 28 * 1 -> 28 * 28 * 3
    SVHN: 28 * 28 * 3
    MNISTM: 28 * 28 * 3
"""

import argparse
import logging
import logging.config
import os
import random
import time

import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import utils
from dann import Class_Classifier, Domain_Classifier, Feature_Extractor

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def val(feature_extractor, class_classifier, domain_classifier, loader, domain_indice, class_criterion, domain_criterion, gen_csv=None):
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    batch_domain_acc = []
    batch_class_acc  = []
    batch_domain_los = []
    batch_class_los  = []

    pred_results = pd.DataFrame()

    constant = opt.alpha
    domain_indice = torch.tensor(domain_indice).type(torch.long).to(DEVICE)

    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for _, (img, label, img_name) in enumerate(loader, 1):
        batch_size   = len(img)

        img, label   = img.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        feature      = feature_extractor(img).view(batch_size, -1)
        class_pred   = class_classifier(feature)
        domain_pred  = domain_classifier(feature, constant)
        domain_label = domain_indice.expand(batch_size)

        # loss
        loss = class_criterion(class_pred, label)
        batch_class_los.append(loss.item() * batch_size)

        loss = domain_criterion(domain_pred, domain_label)
        batch_domain_los.append(loss.item() * batch_size)

        # Class Accuracy
        class_pred = class_pred.cpu().detach().numpy()
        class_label = label.cpu().detach().numpy()
        acc = np.mean(np.argmax(class_pred, axis=1) == class_label)
        batch_class_acc.append(acc * batch_size)

        # Domain Accuracy
        domain_pred = domain_pred.cpu().detach().numpy()
        domain_label = domain_label.cpu().detach().numpy()
        acc = np.mean(np.argmax(domain_pred, axis=1) == domain_label)
        batch_domain_acc.append(acc * batch_size)

        list_of_tuple = list(zip(img_name, class_pred))
        pred_result   = pd.DataFrame(list_of_tuple, columns=["image_name", "label"])
        pred_results  = pd.concat((pred_result, pred_result), axis=0, ignore_index=True)

    #------------------------------------------------------------
    # gen_csv: the path to save the csv, if not None -> Write it!
    #------------------------------------------------------------
    if gen_csv:
        pred_results.to_csv(gen_csv, index=False)
        print("Output File have been written to {}".format(gen_csv))

    domain_acc  = np.sum(batch_domain_acc) / len(loader.dataset)
    class_acc   = np.sum(batch_class_acc) / len(loader.dataset)
    domain_loss = np.sum(batch_domain_los) / len(loader.dataset)
    class_loss  = np.sum(batch_class_los) / len(loader.dataset)

    return class_acc, class_loss, domain_acc, domain_loss

def main():
    os.makedirs(os.path.join(opt.write, ".."), exist_ok=True)

    #-------------------------
    # Construce the DANN model
    #-------------------------
    feature_extractor = Feature_Extractor()
    class_classifier  = Class_Classifier()
    domain_classifier = Domain_Classifier()

    class_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    domain_criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    
    DOMAINS = [("usps", "mnistm"), ("mnistm", "svhn"), ("svhn", 'usps')]
    for SOURCE, TARGET in DOMAINS:
        # Read the DANN model
        feature_extractor, class_classifier, domain_classifier = utils.loadDANN(opt.model, feature_extractor, class_classifier, domain_classifier)
        
        # Create Dataloader
        source_black = True if SOURCE == 'usps' else False
        target_black = True if TARGET == 'usps' else False
        source_test_set  = dataset.NumberClassify("./hw3_data/digits", SOURCE, train=False, black=source_black, transform=transforms.ToTensor())
        target_train_set = dataset.NumberClassify("./hw3_data/digits", TARGET, train=False, black=target_black, transform=transforms.ToTensor())
        target_test_set  = dataset.NumberClassify("./hw3_data/digits", TARGET, train=False, black=target_black, transform=transforms.ToTensor())
        print("Source_test: \t{}, {}".format(SOURCE, len(source_test_set)))
        print("Target_test: \t{}, {}".format(TARGET, len(target_test_set)))    
        source_test_loader  = DataLoader(source_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
        target_train_loader = DataLoader(target_train_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
        target_test_loader  = DataLoader(target_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)

        # Predict
        class_acc, class_loss, domain_acc, domain_loss = val(feature_extractor, class_classifier, domain_classifier, source_test_loader, 0, class_criterion, domain_criterion, gen_csv=None)
        print("Source_Test: ")
        print("[class_acc: {:.2f}] [class_loss: {:.4f}] [domain_acc: {:.2f}] [domain_loss: {:.4f}]".format(100 * class_acc, class_loss, 100 * domain_acc, domain_loss))
        
        class_acc, class_loss, domain_acc, domain_loss = val(feature_extractor, class_classifier, domain_classifier, target_train_loader, 1, class_criterion, domain_criterion, gen_csv=None)
        print("Target_Train: ")
        print("[class_acc: {:.2f}] [class_loss: {:.4f}] [domain_acc: {:.2f}] [domain_loss: {:.4f}]".format(100 * class_acc, class_loss, 100 * domain_acc, domain_loss))
        
        class_acc, class_loss, domain_acc, domain_loss = val(feature_extractor, class_classifier, domain_classifier, target_test_loader, 1, class_criterion, domain_criterion, gen_csv=opt.output)
        print("Target_Test: ")
        print("[class_acc: {:.2f}] [class_loss: {:.4f}] [domain_acc: {:.2f}] [domain_loss: {:.4f}]".format(100 * class_acc, class_loss, 100 * domain_acc, domain_loss))
        
if __name__ == "__main__":
    os.system("clear")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=int, default=0.25, help="Backpropagation constant")
    parser.add_argument("--dataset", type=str, help="The root of input dataset")
    parser.add_argument("--output", type=str, help="Predict file to write")
    parser.add_argument("--threads", type=int, default=8, help="Number of cpu threads to use during batch generation")
    
    subparser    = parser.add_subparsers(dest="test_domain")
    
    svhnparser   = subparser.add_parser("svhn")
    svhnparser.add_argument("--output", help="The predict csvfile path.")

    mnistmparser = subparser.add_parser("mnistm")
    mnistmparser.add_argument("--output", help="The predict csvfile path.")
    
    uspsparser   = subparser.add_parser("usps")
    uspsparser.add_argument("--output", help="The predict csvfile path.")
    
    opt = parser.parse_args()
    print(opt)
    
    main()
