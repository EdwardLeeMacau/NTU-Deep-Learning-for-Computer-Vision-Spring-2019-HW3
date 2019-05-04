"""
  FileName     [ dann.py ]
  PackageName  [ HW3 ]
  Synopsis     [ DANN Models ]

  Dataset:
    USPS: 28 * 28 * 1 -> 28 * 28 * 3
    SVHN: 28 * 28 * 3
    MNISTM: 28 * 28 * 3

  DANN Models:
    Feature Layers
    Class predictor
    Domain predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import transforms

import dataset
import utils

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.alpha
        return grad_output, None

def grad_reverse(x, constant):
    return ReverseLayerF.apply(x, constant)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        self.feature = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),

            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Dropout2d(0.5),
        )

    def forward(self, x):
        x = x.expand(-1, 3, 28, 28)
        x = self.feature(x)
        # print("Feature.shape: \t{}".format(x.shape))

        return x

class Class_Classifier(nn.Module):
    def __init__(self):
        super(Class_Classifier, self).__init__()
        
        self.class_detect = nn.Sequential(
            # Layer 1
            nn.Linear(128 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            # Layer 3
            nn.Linear(2048, 10),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, feature):
        x = self.class_detect(feature)
    
        return x

class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()

        self.domain_detect = nn.Sequential(
            # Layer 1
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 2
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # Layer 3
            nn.Linear(1024, 2),
            # nn.LogSoftmax(dim=1),
        )

    def forward(self, feature, constant):
        # feature = ReverseLayerF.grad_reverse(feature, constant)
        feature = grad_reverse(feature, constant)
        d = self.domain_detect(feature)

        return d

def dann_unittest():
    SOURCE = "usps"
    DEVICE = utils.selectDevice()
    
    feature_extractor = Feature_Extractor().to(DEVICE)
    class_classifier  = Class_Classifier().to(DEVICE)
    domain_classifier = Domain_Classifier().to(DEVICE)

    trainset = dataset.NumberClassify("./hw3_data/digits", SOURCE, train=True, black=True, transform=transforms.ToTensor())
    
    loader = DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)
    
    for _, (img, label, _) in enumerate(loader):
        img, label = img.to(DEVICE), label.to(DEVICE)

        print("Label.shape: \t{}".format(label.shape))

        f = feature_extractor(img)
        print("Feature.shape: \t{}".format(f.shape))

        f = f.view(f.shape[0], -1)
        y = class_classifier(f)
        print("Class_Label.shape: \t{}".format(y.shape))

        d = domain_classifier(f, 0.5)
        print("Domain_Label.shape: \t{}".format(d.shape))

        # Print 1 time only
        break

if __name__ == "__main__":
    dann_unittest()
    print("DANN Unittest Done!")
