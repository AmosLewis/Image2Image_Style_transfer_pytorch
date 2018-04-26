import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms


def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight') and m.weight is not None:
        if m.weight.ndimension() > 1:
            xavier_uniform(m.weight.data)
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()


def format_images(images):
    # convert (n, c, h, w) to a single image grid (1, c, g*h, g*w)
    c = images.size(1)
    h = images.size(2)
    w = images.size(3)
    gridsize = int(math.floor(math.sqrt(images.size(0))))
    images = images[:gridsize * gridsize]  # (g*g, c, h, w)
    images = images.view(gridsize, gridsize, c, h, w)  # (g,g,c,h,w)
    images = images.permute(0, 3, 1, 4, 2).contiguous()  # (g, h, g, w, c)
    images = images.view(1, gridsize * h, gridsize * w, c)  # (1, g*h, g*w, c)
    images = images.permute(0, 3, 1, 2)  # (1, c, g*h, g*w)
    return images


def save_args(args):
    # Save argparse arguments to a file for reference
    os.makedirs(args.save_directory, exist_ok=True)
    with open(os.path.join(args.save_directory, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write("{}={}\n".format(k, v))


class Reshape(nn.Module):
    # Module that just reshapes the input
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class GANModel(nn.Module):
    # GAN containing generator and discriminator
    def __init__(self, args, discriminator, generator):
        super(GANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = args.latent_dim
        self._state_hooks = {}  # used by inferno for logging
        self.apply(initializer)  # initialize the parameters

    def generate(self, latent):
        # Generate fake images from latent inputs
        xfake = self.generator(latent)
        # Save images for later
        self._state_hooks['xfake'] = xfake
        self._state_hooks['generated_images'] = format_images(xfake)  # log the generated images
        return xfake

    def discriminate(self, x):
        # Run discriminator on an input
        return self.discriminator(x)

    def y_fake(self, latent):
        # Run discriminator on generated images
        yfake = self.discriminate(self.generate(latent))
        return yfake

    def y_real(self, xreal):
        # Run discriminator on real images
        yreal = self.discriminate(xreal)
        # Save images for later
        self._state_hooks['xreal'] = xreal
        self._state_hooks['real_images'] = format_images(xreal)
        return yreal

    def latent_sample(self, xreal):
        # Generate latent samples of same shape as real data
        latent = xreal.data.new(xreal.size(0), self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        return latent

    def forward(self, xreal):
        # Calculate and return y_real and y_fake
        return self.y_real(xreal), self.y_fake(self.latent_sample(xreal))


class GeneratorTrainingCallback(Callback):
    # Callback periodically trains the generator
    def __init__(self, args, parameters, criterion):
        self.criterion = criterion
        self.opt = Adam(parameters, args.generator_lr)
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.count = 0
        self.frequency = args.generator_frequency

    def end_of_training_iteration(self, **_):
        # Each iteration check if it is time to train the generator
        self.count += 1
        if self.count > self.frequency:
            self.train_generator()
            self.count = 0

    def train_generator(self):
        # Train the generator
        # Generate latent samples
        if self.trainer.is_cuda():
            latent = torch.cuda.FloatTensor(self.batch_size, self.latent_dim)
        else:
            latent = torch.FloatTensor(self.batch_size, self.latent_dim)
        torch.randn(*latent.size(), out=latent)
        latent = Variable(latent)
        # Calculate yfake
        yfake = self.trainer.model.y_fake(latent)
        # Calculate loss
        loss = self.criterion(yfake)
        # Perform update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class GenerateDataCallback(Callback):
    # Callback saves generated images to a folder
    def __init__(self, args, gridsize=10):
        super(GenerateDataCallback, self).__init__()
        self.count = 0  # iteration counter
        self.image_count = 0  # image counter
        self.frequency = args.image_frequency
        self.gridsize = gridsize
        self.latent = torch.randn(gridsize * gridsize, args.latent_dim)

    def end_of_training_iteration(self, **_):
        # Check if it is time to generate images
        self.count += 1
        if self.count > self.frequency:
            self.save_images()
            self.count = 0

    def generate(self, latent):
        # Set eval, generate, then set back to train
        self.trainer.model.eval()
        generated = self.trainer.model.generate(Variable(latent))
        self.trainer.model.train()
        return generated

    def save_images(self):
        # Generate images
        path = os.path.join(self.trainer.save_directory, 'generated_images')
        os.makedirs(path, exist_ok=True)  # create directory if necessary
        image_path = os.path.join(path, '{:08d}.png'.format(self.image_count))
        self.image_count += 1
        # Copy latent to cuda if necessary
        if self.trainer.is_cuda():
            latent = self.latent.cuda()
        else:
            latent = self.latent
        generated = self.generate(latent)
        # Reshape, scale, and cast the data so it can be saved
        grid = format_images(generated).squeeze(0).permute(1, 2, 0)
        if grid.size(2) == 1:
            grid = grid.squeeze(2)
        array = grid.data.cpu().numpy() * 255.
        array = array.astype(np.uint8)
        # Save the image
        Image.fromarray(array).save(image_path)
