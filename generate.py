"""
  FileName     [ generate.py ]
  PackageName  [ HW3 ]
  Synopsis     [ Generate images from GAN / ACGAN. ]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

import utils
from torchvision.utils import save_image

# DEVICE = utils.selectDevice()
DEVICE = "cpu"

np.random.seed(0)
torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], 3, 64, 64)
        return img

class ACGAN_Generator(nn.Module):
    def __init__(self):
        super(ACGAN_Generator, self).__init__()

        self.linear = nn.Linear(102, 512 * 4 * 4)
        self.bn0    = nn.BatchNorm2d(512)
        self.relu0  = nn.ReLU(inplace=True)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        # ---------------------------------------------
        #  gen: 
        #    [z labels]: z(100), label(2) to show that 
        #    [1 0] = - (negative feature)
        #    [0 1] = + (positive feature)
        #    Ref: hw3.dataset.py
        # ---------------------------------------------
        gen = torch.cat((z, labels), dim=1)
        gen = self.linear(gen).view(-1, 512, 4, 4)
        gen = self.bn0(gen)
        gen = self.relu0(gen)
        img = self.conv_blocks(gen)
        return img

def main(command):
    os.system("clear")

    acgan = command == 'acgan'
    dcgan = command == 'dcgan'

    if acgan:
        generator = utils.loadModel(opt.model, ACGAN_Generator())
    
    if dcgan:
        generator = utils.loadModel(opt.model, DCGAN_Generator())

    if acgan:
        feature = utils.faceFeatures[7]
        print("Using Feature: {}".format(feature))

        z = torch.Tensor(np.random.normal(0, 1, size=(10, opt.latent_dim, 1, 1)), dtype=torch.float)
        z = torch.cat((z, z), dim=0)
        labels = torch.Tensor(np.array([[1 - num, num] for num in range(2) for _ in range(10)]), dtype=torch.float)
        
        gen_imgs = generator(z, labels)
        save_image(gen_imgs.data, os.path.join(opt.output, "fig2_2.jpg"), nrow=10, normalize=True)
    
    if dcgan:
        z = np.random.normal(0, 1, size=(32, opt.latent_dim, 1, 1))
        gen_imgs = generator(z)

        save_image(gen_imgs.data, os.path.join(opt.output, "fig1_2,jpg", nrow=8, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=100, help="The length of z")
    parser.add_argument("--channels", type=int, default=3, help="The number of channels, default 3 (RGB)")

    subgroup = parser.add_subparsers(dest="command", help="Use ACGAN / DCGAN?")
    
    acganparser = subgroup.add_parser('acgan', help="using acgan generation code.")
    acganparser.add_argument("--model", type=str, default="./models/acgan/20190503-Male/generator_30.pth", help="The model to read.")
    acganparser.add_argument("--output", type=str, help="The output path of the images")
    
    dcganparser = subgroup.add_parser('dcgan', help="using dcgan generation code.")
    dcganparser.add_argument("--model", type=str, default="./models/dcgan/20190501/generator_20.pth", help="The model to read.")
    dcganparser.add_argument("--output", type=str, help="The output path of the images")
    
    opt = parser.parse_args()

    main(opt.command)
