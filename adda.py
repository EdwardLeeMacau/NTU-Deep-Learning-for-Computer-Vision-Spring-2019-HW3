"""
  FileName     [ adda.py ]
  PackageName  [ HW3 ]
  Synopsis     [ ADDA Models ]

  Dataset:
    USPS: 28 * 28 * 1 -> 28 * 28 * 3
    SVHN: 28 * 28 * 3
    MNISTM: 28 * 28 * 3

  DANN Models:
    Source CNN
    Target CNN
    Discriminator
    Class predictor
    Domain predictor
"""

import torch.nn.functional as F
from torch import nn

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        
        self.restored = False

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),


            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

    def forward(self, img):
        feature = self.encoder(img)
        print("Feature.shape: {}".format(feature.shape))

        return feature

class Classifier(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Classifier, self).__init__()
        
        #-------------------------------------
        # Train the model with DANN strategic
        #   SOURCE -> TARGET
        #-------------------------------------
        self.classify = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_dims, output_dims),

            # Using CrossEntropy to train this layer.
            # nn.LogSoftmax()
        )

    def forward(self, feature):
        x = self.classify(feature)
        print("Output.shape: {}".format(x.shape))

        return x

class Discriminator(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(Discriminator, self).__init__()

        self.restored = False

        #-------------------------------------
        # Train the model with DANN strategic
        #   SOURCE -> TARGET
        #-------------------------------------
        self.classify = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dims, output_dims),

            # Using CrossEntropy to train this layer.
            # nn.LogSoftmax()
        )

    def forward(self, feature):
        """Forward the discriminator."""
        out = self.classify(feature)
        return out