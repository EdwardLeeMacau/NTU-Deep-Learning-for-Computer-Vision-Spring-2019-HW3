import sys

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn, optim

faceFeatures = [
    "Bangs", "Big_Lips", "Black_Hair", "Blond_Hair", "Brown_Hair", 
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Smiling", 
    "Straight_Hair", "Wavy_Hair", "Wearing_Lipstick"
]

def set_optimizer_lr(optimizer, lr):
    """ set the learning rate in an optimizer, without rebuilding the whole optimizer """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer

def selectDevice():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return device

def saveCheckpoint(checkpoint_path, model, optimizer, scheduler, epoch):
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, checkpoint_path)
    # print('model saved to %s' % checkpoint_path)

def loadCheckpoint(checkpoint_path: str, model: nn.Module, optimizer: optim, scheduler: optim.lr_scheduler.MultiStepLR):
    state = torch.load(checkpoint_path)
    resume_epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    print('model loaded from %s' % checkpoint_path)

    return model, optimizer, resume_epoch, scheduler

def saveModel(checkpoint_path: str, model: nn.Module):
    state = {
        'state_dict': model.state_dict(),
    }
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def loadModel(checkpoint_path: str, model: nn.Module):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('Model loaded from %s' % checkpoint_path)

    return model

def checkpointToModel(checkpoint_path: str, model_path: str):
    state = torch.load(checkpoint_path)

    newState = {
        'state_dict': state['state_dict']
    }

    torch.save(newState, model_path)