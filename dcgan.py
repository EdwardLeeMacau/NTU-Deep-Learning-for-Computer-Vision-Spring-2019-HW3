"""
  FileName     [ gan.py ]
  PackageName  [ HW3 ]
  Synopsis     [ DCGAN Model, train method and generate method. ]

  Dataset: CelebA
  Input.shape: 64 * 64 * 3
"""

import argparse
import math
import os
import logging
import logging.config

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from matplotlib import pyplot as plt

import dataset
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--tag", type=str, help="name of the model")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
parser.add_argument("--log_interval", type=int, default=10, help="interval between everytime logging the G/D loss.")
parser.add_argument("--save_interval", type=int, default=1000, help="interval to save the models")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
device = utils.selectDevice()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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
            
            nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(opt.channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss().to(device)

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataset = dataset.CelebA("./hw3_data/face/", utils.faceFeatures[0], transform=transforms.Compose([transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

torch.set_default_dtype(torch.float)

def train(epoch, noise_threshold=30):
    generator_loss = []
    discriminator_loss = []

    for i, (imgs, _) in enumerate(dataloader, 1):

        # Adversarial ground truths
        # if epoch > noise_threshold:
        valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        fake  = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)
        # else:
        # valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).to(device)
        # fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).to(device)

        # Configure input
        real_imgs = imgs.type(torch.float).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.Tensor(np.random.normal(0, 1, size=(imgs.shape[0], opt.latent_dim, 1, 1))).to(device)

        # Generate a batch of images
        gen_imgs = generator(z)
        # print('imgs.shape: \t{}'.format(gen_imgs.shape))

        # Loss measures generator's ability to fool the discriminator
        # print('valid.shape: \t{}'.format(valid.shape))
        # print('discr.shape: \t{}'.format(discriminator(gen_imgs).view(imgs.shape[0], -1).shape))
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        generator_loss.append(g_loss.item())

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        discriminator_loss.append(d_loss.item())

        batches_done = (epoch - 1) * len(dataloader) + i

        # --------------------------
        #  Logging, sampling, saving
        # --------------------------
        # 1. Logging
        if batches_done % opt.log_interval == 0:
            print("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, i, len(dataloader), d_loss.item(), g_loss.item()))
        
        # 2. Saving
        if batches_done % opt.save_interval == 0:
            number   = batches_done // opt.save_interval

            savepath = "./models/gan/{}".format(opt.tag)
            utils.saveModel(os.path.join(savepath, "generator_{}.pth".format(number)), generator)
            utils.saveModel(os.path.join(savepath, "discriminator_{}.pth".format(number)), discriminator)
            
            print("Model saved to: {}, iteration: {}".format(savepath, number))

        # 3. Sampling
        if batches_done % opt.sample_interval == 0:
            number     = batches_done // opt.sample_interval
            samplepath = "./output/gan/{}/{}.png".format(opt.tag, number)
            save_image(gen_imgs.data[:32], samplepath, nrow=8, normalize=True)
            print("Image saved to: {}".format(samplepath))

    return generator_loss, discriminator_loss

def generate(generator, num_imgs):
    """ Use trained generator to generate images, save it. """
    # Fixed the random seed.
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    z = np.random.normal(0, 1, size=(num_imgs, opt.latent_dim, 1, 1)).to(device)
    gen_imgs = generator(z)

    save_image(gen_imgs.data, "./output/gan/hw1.png", nrow=8, normalize=True)

def main():
    os.system("clear")
    print(opt)

    os.makedirs("./output", exist_ok=True)
    os.makedirs("./output/gan", exist_ok=True)
    os.makedirs("./output/gan/{}".format(opt.tag), exist_ok=True)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/gan", exist_ok=True)
    os.makedirs("./models/gan/{}".format(opt.tag), exist_ok=True)

    g_loss_iteration = []
    d_loss_iteration = []
    
    for epoch in range(1, opt.n_epochs):
        generator_loss, discriminator_loss = train(epoch)
        g_loss_iteration += generator_loss
        d_loss_iteration += discriminator_loss

        # Only Keep the last 5 epochs loss
        if epoch > 5:
            threshold = -5 * len(dataloader)
            
            g_loss_iteration, d_loss_iteration = g_loss_iteration[threshold: ], d_loss_iteration[threshold: ]
            x = np.arange(start=((epoch - 5) * len(dataloader) + 1), stop=(epoch * len(dataloader) + 1))
        else:
            x = np.arange(start=1, stop=(epoch * len(dataloader) + 1))

        plt.clf()
        plt.figure(figsize=(12.8, 7.2))
        plt.plot(x, g_loss_iteration, label='G loss', color='r', linewidth=0.25)
        plt.plot(x, d_loss_iteration, label='D loss', color='b', linewidth=0.25)
        plt.legend(loc=0)
        plt.ylabel("Loss")
        plt.xlabel("Iteration(s)")
        plt.title("G / D Loss vs Iteration(s)")
        plt.savefig("DCGAN_Loss_Iteration.png")

    # generate(generator, 32)

if __name__ == "__main__":
    main()
