"""
  FileName     [ train_adda.py ]
  PackageName  [ HW3 ]
  Synopsis     [ ADDA training methods ]

  Dataset:
    USPS: 28 * 28 * 1 -> 28 * 28 * 3
    SVHN: 28 * 28 * 3
    MNISTM: 28 * 28 * 3
"""

import argparse
import datetime
import logging
import logging.config
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset
import predict
import utils
from adda import Feature, Classifier, Discriminator
from dann import Class_Classifier, Domain_Classifier, Feature_Extractor

# Set as true when the I/O shape of the model is fixed
cudnn.benchmark = True
DEVICE = utils.selectDevice()

def train_source(encoder, classifier, optimizer, loader, criterion, epoch):
    encoder.train()
    classifier.train()

    for index, (img, label, _) in enumerate(loader, 1):
        #-------------------------------
        # Prepare the images and labels
        #   img, label
        #-------------------------------
        img, label = img.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        
        #-------------------------------------------
        # Setup optimizer
        # Prepare the learning rate
        #-------------------------------------------
        # optim = utils.set_optimizer_lr(optim, p)
        optimizer.zero_grad()

        #-------------------------------
        # Get features, class pred
        #-------------------------------
        feature = encoder(img).view(-1, 128 * 7 * 7)
        class_predict = classifier(feature)
        
        #---------------------------------------
        # Compute the loss
        #------------------------------------
        # print(class_predict)
        # print(label)
        loss = criterion(class_predict, label)
        loss.backward()
        optimizer.step()

        class_predict = class_predict.cpu().detach().numpy()
        label = label.cpu().detach().numpy()

        acc = np.mean(np.argmax(class_predict, axis=1) == label)

        if index % opt.log_interval == 0:
            print("[Epoch {}] [ {:4d}/{:4d} ] [acc: {:.2f}%] [loss: {:.4f}]".format(
                    epoch, index, len(loader), 100 * acc, loss.item()))

    return encoder, classifier

def train_target(source_encoder, target_encoder, discriminator, criterion,
                 source_loader, target_loader, d_optimizer, e_optimizer, epoch):
    target_encoder.train()
    discriminator.train()

    for index, ((source_img, _, _), (target_img, _, _)) in enumerate(zip(source_loader, target_loader), 1):
        # --------------
        # Prepare labels
        # --------------
        source_img, target_img = source_img.to(DEVICE), target_img.to(DEVICE)
        batch_source_size = source_img.shape[0]
        batch_target_size = target_img.shape[0]

        source_label = torch.zeros(batch_source_size).type(torch.long).to(DEVICE)
        target_label = torch.ones(batch_target_size).type(torch.long).to(DEVICE)

        # ------------------------
        # Train discriminator 
        # ------------------------
        d_optimizer.zero_grad()
        e_optimizer.zero_grad()

        # extract and concat features
        source_feature = source_encoder(source_img)
        target_feature = target_encoder(target_img)
        feature = torch.cat((source_feature, target_feature), 0)

        # predict on discriminator
        domain_pred = discriminator(feature)

        # prepare real and fake label

        domain_label = torch.cat((source_label, target_label), dim=0)

        # compute loss for critic
        discriminator_loss = criterion(domain_pred, domain_label)
        discriminator_loss.backward()
        d_optimizer.step()

        domain_pred = torch.squeeze(feature.max(1)[1])
        domain_acc  = (domain_pred == domain_label).float().mean()

        # ---------------------------
        # Train target encoder 
        # ---------------------------

        # zero gradients for optimizer
        d_optimizer.zero_grad()
        e_optimizer.zero_grad()

        # extract and target features
        target_feature = target_encoder(target_img)

        # predict on discriminator
        domain_pred  = discriminator(target_feature)
        encoder_loss = criterion(domain_pred, source_label)
        encoder_loss.backward()
        e_optimizer.step()

        if index % opt.log_interval == 0:
                print("[Epoch {}] [ {:4d}/{:4d} ] [domain_acc: {:.2f}] [loss_D: {:.4f}] [loss_E: {:.4f}]".format(
                        epoch, index, min(len(source_loader), len(target_loader)), 100 * domain_acc, discriminator_loss.item(), encoder_loss.item()))
    
    return target_encoder, discriminator

def val(encoder, classifier, loader, class_criterion):
    encoder.eval()
    classifier.eval()
    
    batch_class_acc = []
    batch_class_los = []

    #----------------------------
    # Calculate the accuracy, loss
    #----------------------------
    for _, (img, label, _) in enumerate(loader, 1):
        batch_size   = len(img)

        img, label   = img.to(DEVICE), label.type(torch.long).view(-1).to(DEVICE)
        feature      = encoder(img).view(batch_size, -1)
        class_pred   = classifier(feature)
        
        # loss
        loss = class_criterion(class_pred, label)
        batch_class_los.append(loss.item() * batch_size)

        # Class Accuracy
        class_pred = class_pred.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        acc = np.mean(np.argmax(class_pred, axis=1) == label)
        batch_class_acc.append(acc * batch_size)

    class_acc  = np.sum(batch_class_acc) / len(loader.dataset)
    class_loss = np.sum(batch_class_los) / len(loader.dataset)

    return class_acc, class_loss

def draw_graphs(x, y_source, y_target, threshold, source_epochs, source, target):
    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, y_source[:, 0], 'r', label='Source Class Accuracy', linewidth=1)
    plt.plot(x, y_target[:, 0], 'b', label='Target Class Accuracy', linewidth=1)
    plt.plot(x, np.asarray([threshold]).repeat(len(x)), 'b-', linewidth=0.1)    
    plt.plot([source_epochs, source_epochs], [0.0, 1.0], 'b-', linewidth=0.1)
    plt.legend(loc=0)
    plt.xlabel("Epochs(s)")
    plt.title("[Acc - Train {} Test {}] vs Epoch(s)".format(source, target))
    plt.savefig("ADDA_{}_{}-Acc.png".format(source, target))
    plt.close()

    plt.clf()
    plt.figure(figsize=(12.8, 7.2))
    plt.plot(x, y_source[:, 1], 'r', label='Source Class Loss', linewidth=1)
    plt.plot(x, y_target[:, 1], 'b', label='Target Class Loss', linewidth=1)
    plt.legend(loc=0)
    plt.xlabel("Epochs(s)")
    plt.title("[Loss - Train {} Test {}] vs Epoch(s)".format(source, target))
    plt.savefig("ADDA_{}_{}-Loss.png".format(source, target))
    plt.close()

    return

def adversarial_discriminative_domain_adaptation(source, target, source_epochs, target_epochs, threshold, source_lr, target_lr, weight_decay):
    """ Using ADDA framework to train the network. """
    #-----------------------------------------------------
    # Create Model, optimizer, scheduler, and loss function
    #------------------------------------------------------
    source_encoder   = Feature().to(DEVICE)
    target_encoder   = Feature().to(DEVICE)
    class_classifier = Classifier(128 * 7 * 7, 1000, 10).to(DEVICE)
    discriminator    = Discriminator(128 * 7 * 7, 500, 2).to(DEVICE)

    source_optimizer = optim.Adam([{'params': source_encoder.parameters()},
                                   {'params': class_classifier.parameters()}], 
                                    lr=source_lr, betas=(opt.b1, opt.b2), weight_decay=weight_decay)
    
    source_scheduler = optim.lr_scheduler.MultiStepLR(source_optimizer, milestones=[], gamma=0.1)
    
    encoder_criterion     = nn.CrossEntropyLoss().to(DEVICE)
    adversarial_criterion = nn.BCELoss().to(DEVICE)
    
    #------------------
    # Create Dataloader
    #------------------
    source_black = True if source == 'usps' else False
    target_black = True if target == 'usps' else False
    
    source_train_set = dataset.NumberClassify("./hw3_data/digits", source, train=True, black=source_black, transform=transforms.ToTensor())
    target_train_set = dataset.NumberClassify("./hw3_data/digits", target, train=True, black=target_black, transform=transforms.ToTensor())
    source_test_set  = dataset.NumberClassify("./hw3_data/digits", source, train=False, black=source_black, transform=transforms.ToTensor())
    target_test_set  = dataset.NumberClassify("./hw3_data/digits", target, train=False, black=target_black, transform=transforms.ToTensor())
    
    print("Source_train: \t{}, {}".format(source, len(source_train_set)))
    print("Source_test: \t{}, {}".format(source, len(source_test_set)))
    print("Target_train: \t{}, {}".format(target, len(target_train_set)))
    print("Target_test: \t{}, {}".format(target, len(target_test_set)))
    
    source_train_loader = DataLoader(source_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    source_test_loader  = DataLoader(source_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    target_train_loader = DataLoader(target_train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
    target_test_loader  = DataLoader(target_test_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.threads)
    
    source_pred_values = []
    target_pred_values = []

    #------------------
    # Train Source Domain
    #------------------
    if not opt.pretrain:
        for epoch in range(1, source_epochs + 1):
            source_scheduler.step()
            source_encoder, class_classifier = train_source(source_encoder, class_classifier, source_optimizer, source_train_loader, encoder_criterion, epoch)

            source_acc, source_loss = val(source_encoder, class_classifier, source_test_loader, encoder_criterion)
            target_acc, target_loss = val(source_encoder, class_classifier, target_test_loader, encoder_criterion)
            # Return: class_acc, class_loss, domain_acc, domain_loss

            print("[Epoch {}] [ src_acc: {:.2f}% ] [ tgt_acc: {:.2f}% ] [ src_loss: {:.4f} ] [ tgt_loss: {:.4f} ]".format(
                epoch, 100 * source_acc, 100 * target_acc, source_loss, target_loss))

            # Tracing the accuracy, loss
            source_pred_values.append((source_acc, source_loss))
            target_pred_values.append((target_acc, target_loss))

            y_source = np.asarray(source_pred_values, dtype=float)
            y_target = np.asarray(target_pred_values, dtype=float)
            x = np.arange(start=1, stop=epoch + 1)

            # Draw graphs
            draw_graphs(x, y_source, y_target, threshold, source_epochs, source, target)

            with open('statistics.txt', 'a') as textfile:
                textfile.write(datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S"))
                textfile.write(str(source_acc))
                textfile.write(str(target_acc))

            if target_acc > threshold:
                # Update the threshold
                threshold = target_acc
                
                savepath = "./models/adda/{}/ADDA_{}_{}_{}.pth".format(opt.tag, source, target, epoch)
                utils.saveADDA(savepath, source_encoder, target_encoder, class_classifier)
                print("Model saved to: {}".format(savepath))
    
        #-----------------------------------
        # Save the model in the last epochs
        #----------------------------------
        savepath = "./models/adda/{}/ADDA_{}_{}_{}.pth".format(opt.tag, source, target, opt.source_epochs)
        utils.saveADDA(savepath, source_encoder, target_encoder, class_classifier)
        print("Model saved to: {}".format(savepath))

    if opt.pretrain:
        if not os.path.exists(opt.pretrain):
            raise IOError
        
        source_encoder = utils.loadModel(opt.pretrain, source_encoder)

    #---------------------------------
    # Initial the target domain encoder
    #----------------------------------
    target_encoder.load_state_dict(source_encoder.state_dict())
    target_optimizer = optim.Adam(target_encoder.parameters(), target_lr, betas=(opt.b1, opt.b2))
    discri_optimizer = optim.Adam(discriminator.parameters(), target_lr)
    target_scheduler = optim.lr_scheduler.MultiStepLR(target_optimizer, milestones=[], gamma=0.1)

    #------------------
    # Train Target Domain
    #------------------
    for epoch in range(source_epochs + 1, source_epochs + target_epochs + 1):
        target_scheduler.step()
        
        target_encoder, discriminator = train_target(source_encoder, target_encoder, discriminator, adversarial_criterion,
                                                     source_train_loader, target_train_loader, discri_optimizer, target_optimizer, epoch)
        
        source_acc, source_loss = val(source_encoder, class_classifier, source_test_loader, encoder_criterion)
        target_acc, target_loss = val(target_encoder, class_classifier, target_test_loader, encoder_criterion)
        # Return: class_acc, class_loss, domain_acc, domain_loss

        print("[Epoch {}] [ src_acc: {:.2f}% ] [ tgt_acc: {:.2f}% ] [ src_loss: {:.4f} ] [ tgt_loss: {:.4f} ]".format(
               epoch, 100 * source_acc, 100 * target_acc, source_loss, target_loss))

        # Tracing the accuracy, loss
        source_pred_values.append((source_acc, source_loss))
        target_pred_values.append((target_acc, target_loss))

        y_source = np.asarray(source_pred_values, dtype=float)
        y_target = np.asarray(target_pred_values, dtype=float)
        x = np.arange(start=1, stop=epoch+1)
        
        # Draw the graphs
        draw_graphs(x, y_source, y_target, threshold, source_epochs, source, target)

        with open('statistics.txt', 'a') as textfile:
            textfile.write(datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S"))
            textfile.write(str(source_acc))
            textfile.write(str(target_acc))

        if target_acc > threshold:
            # Update the threshold
            threshold = target_acc
            
            savepath = "./models/adda/{}/ADDA_{}_{}_{}.pth".format(opt.tag, source, target, epoch)
            utils.saveADDA(savepath, source_encoder, target_encoder, class_classifier)
            print("Model saved to: {}".format(savepath))

    #-----------------------------------
    # Save the model in the last epochs
    #----------------------------------
    savepath = "./models/adda/{}/ADDA_{}_{}_{}.pth".format(opt.tag, source, target, epoch)
    utils.saveADDA(savepath, source_encoder, target_encoder, class_classifier)
    print("Model saved to: {}".format(savepath))

    return source_encoder, target_encoder, class_classifier, discriminator

def main():
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/adda", exist_ok=True)
    os.makedirs("./models/adda/{}".format(opt.tag), exist_ok=True)
    
    #-------------------------------------
    # Train the model with DANN strategic
    #   SOURCE -> TARGET
    #-------------------------------------
    DOMAINS = [("usps", "mnistm"), ("mnistm", "svhn"), ("svhn", 'usps')]
    
    for SOURCE, TARGET in DOMAINS:
        
        if TARGET == "mnistm":
            threshold = 0.3
        elif TARGET == "svhn":
            threshold = 0.4
        elif TARGET == "usps":
            threshold = 0.3
        else:
            raise NotImplementedError

        adversarial_discriminative_domain_adaptation(SOURCE, TARGET, opt.source_epochs, opt.target_epochs, threshold, 
                                                     source_lr=opt.source_lr, target_lr=opt.target_lr, weight_decay=opt.weight_decay)

if __name__ == "__main__":
    os.system("clear")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_epochs", type=int, default=50, help="number of epochs of training source")
    parser.add_argument("--target_epochs", type=int, default=50, help="number of epochs of training target")
    parser.add_argument("--pretrain", type=str, help="Pretrain source encoder and classifier")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--source_lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--target_lr", type=float, default=2e-5, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight regularization")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--threads", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--tag", type=str, help="name of the model")
    parser.add_argument("--save_interval", type=int, default=10, help="interval epoch between everytime saving the model.")
    parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the training status.")
    
    opt = parser.parse_args()
    print(opt)
    
    main()
