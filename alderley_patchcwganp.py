import argparse
import math
import os
import sys
import numpy as np
from PIL import Image

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from inferno.trainers.callbacks.base import Callback

from gan_utils import Reshape, format_images
from gan_utils import save_args, initializer
from wgan_loss import CWGANDiscriminatorLoss, WGANGeneratorLoss


class AlderleyWrapper(Dataset):
    def __init__(self):
        super(AlderleyWrapper, self).__init__()
        path = "../data/alderley.npy"
        data = np.load(path)
        data = np.transpose(data, (1, 0, 2, 3, 4))
        self.days = torch.Tensor(data[0])
        self.nights = torch.Tensor(data[1])
        print(self.days[0].mean(), self.nights[0].mean())
        print("Alderley dataset loaded")

    def __len__(self):
        return len(self.days)

    def __getitem__(self, item):
        # print(item)
        x, y = self.nights[item], self.days[item]
        # print(x.size())
        # print(y.size())
        return x, y, y


def alderley_cgan_data_loader(args, dataset):
    # Create DataLoader for Alderley
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader

# todo


class CGeneratorNetwork(nn.Sequential):
    # Network for generation
    # Input is (N, 1, 256, 512)
    def __init__(self, args):
        super(CGeneratorNetwork, self).__init__(*[m for m in [
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # N, 32,256,512
            nn.InstanceNorm2d(32) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # N, 64,128,256
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # N, 64,64,128
            nn.InstanceNorm2d(64) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # N, 128,16,32
            nn.InstanceNorm2d(128) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # N, 256,4,8
            nn.InstanceNorm2d(256) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(256, 128, kernel_size=8, stride=4, padding=2),  # N, 128,16,32
            nn.InstanceNorm2d(128) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128, 64, kernel_size=8, stride=4, padding=2),  # N, 64,64,128
            nn.InstanceNorm2d(64) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=8, stride=4, padding=2),  # N, 32,256,512
            nn.InstanceNorm2d(32) if args.generator_instancenorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # N, 3,256,512
            nn.Sigmoid()] if m is not None])

# todo


class patchCDiscriminatorNetwork(nn.Module):
    # Network for discrimination
    # Input is (N, 6, 256, 512)
    def __init__(self, args):
        super(patchCDiscriminatorNetwork, self).__init__()
        self.trunk = nn.Sequential(*[m for m in [
            nn.Conv2d(6, 64, kernel_size=8, stride=4, padding=2),  # N, 64, 64, 128
            # nn.InstanceNorm2d(64) if args.discriminator_instancenorm else None,
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # N, 128, 32, 64
            nn.InstanceNorm2d(128) if args.discriminator_instancenorm else None,
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # N, 256, 16, 32
            nn.InstanceNorm2d(256) if args.discriminator_instancenorm else None,
            nn.LeakyReLU(0.02),
            # todo
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),  # N, 128, 8, 16
            nn.InstanceNorm2d(128) if args.discriminator_instancenorm else None,
            nn.LeakyReLU(0.02),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),  # N, 64, 4, 8
            nn.InstanceNorm2d(64) if args.discriminator_instancenorm else None,
            nn.LeakyReLU(0.02),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # N, 1, 4, 8
            # todo
            Reshape(-1, 1*4*8),
            nn.Sigmoid()] if m is not None
        ])  # N

    def forward(self, x, y):
        h = torch.cat((x, y), dim=1)
        # print(h.size())
        h = self.trunk(h)
        # print(h.size())
        # print(h.mean(dim=1).size())
        return h.mean(dim=1)


class patchCWGANModel(nn.Module):
    # GAN containing generator and discriminator
    def __init__(self, args, discriminator, generator):
        super(patchCWGANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

        self._state_hooks = {}  # used by inferno for logging
        self.apply(initializer)  # initialize the parameters

    def generate(self, y):
        # Generate fake images from input

        xfake = self.generator(y)
        # Save images for later
        self._state_hooks['xfake'] = xfake
        self._state_hooks['y'] = y
        self._state_hooks['generated_images'] = format_images(xfake)  # log the generated images
        return xfake

    def discriminate(self, x, y):
        # Run discriminator on an input
        return self.discriminator(x, y)

    def y_fake(self, y):
        # Run discriminator on generated images
        yfake = self.discriminate(self.generate(y), y)
        return yfake

    def y_real(self, xreal, y):
        # Run discriminator on real images
        yreal = self.discriminate(xreal, y)
        # Save images for later
        self._state_hooks['xreal'] = xreal
        self._state_hooks['real_images'] = format_images(xreal)
        return yreal

    def forward(self, xreal, y):
        # Calculate and return y_real and y_fake

        return self.y_real(xreal, y), self.y_fake(y)

# class CWGANDiscriminatorLoss(WGANDiscriminatorLoss):
#     def discriminate(self, xmix):
#         y = self.model._state_hooks['y']
#         return self.model.discriminate(xmix, y)


class CGenerateDataCallback(Callback):
    # Callback saves generated images to a folder

    def __init__(self, args, dataset, gridsize=1):
        super(CGenerateDataCallback, self).__init__()
        self.count = 0  # iteration counter
        self.image_count = 0  # image counter
        self.frequency = args.image_frequency
        self.gridsize = gridsize
        self.dataset = dataset
        self.y = self.dataset[0][1].unsqueeze(0)
        self.xreal = self.dataset[0][0].unsqueeze(0)

    def end_of_training_iteration(self, **_):
        # Check if it is time to generate images
        if(self.image_count == 0 and self.count == 0):
            self.save_images(source=True)
            self.save_images(real=True)
        self.count += 1
        if self.count > self.frequency:
            self.save_images()
            self.count = 0

    def generate(self):
        # Set eval, generate, then set back to train
        self.trainer.model.eval()
        y = Variable(self.y)
        if self.trainer.is_cuda():
            y = y.cuda()
        generated = self.trainer.model.generate(y)
        self.trainer.model.train()
        return generated

    def save_images(self, source=False, real=False):
        # Generate images
        path = os.path.join(self.trainer.save_directory, 'generated_images')
        os.makedirs(path, exist_ok=True)  # create directory if necessary
        self.image_count += 1
        if(source):
            image = Variable(self.y)
            image_path = os.path.join(path, 'source.png'.format(self.image_count))
        elif(real):
            image = Variable(self.xreal)
            image_path = os.path.join(path, 'real_target.png'.format(self.image_count))
        else:
            image = self.generate()
            image_path = os.path.join(path, 'target_generated{:08d}.png'.format(self.image_count))
        # Reshape, scale, and cast the data so it can be saved
        grid = format_images(image).squeeze(0).permute(1, 2, 0)
        if grid.size(2) == 1:
            grid = grid.squeeze(2)
        array = grid.data.cpu().numpy() * 255.
        array = array.astype(np.uint8)
        # Save the image
        Image.fromarray(array).save(image_path)


class CGeneratorTrainingCallback(Callback):
    # Callback periodically trains the generator
    def __init__(self, args, parameters, criterion, dataset):
        self.criterion = criterion
        self.opt = Adam(parameters, args.generator_lr)
        self.batch_size = args.batch_size
        self.count = 0
        self.frequency = args.generator_frequency
        self.dataset = dataset
        self.len = len(self.dataset)

    def end_of_training_iteration(self, **_):
        # Each iteration check if it is time to train the generator
        self.count += 1
        if self.count > self.frequency:
            self.train_generator()
            # TODO : add argument to callback
            self.count = 0

    def train_generator(self):
        # Train the generator

        # Calculate yfake
        y = Variable(self.dataset[np.random.randint(0, self.len)][1]).unsqueeze(0)
        if self.trainer.is_cuda():
            y = y.cuda()
        yfake = self.trainer.model.y_fake(y)
        # Calculate loss
        loss = self.criterion(yfake)
        # Perform update
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def run(args):
    dataset = AlderleyWrapper()

    save_args(args)  # save command line to a file for reference
    train_loader = alderley_cgan_data_loader(args, dataset=dataset)  # get the data
    # todo
    model = patchCWGANModel(
        args,
        discriminator=patchCDiscriminatorNetwork(args),
        generator=CGeneratorNetwork(args))

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(CWGANDiscriminatorLoss(penalty_weight=args.penalty_weight, model=model))
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)

    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(args.epochs)
    trainer.register_callback(CGenerateDataCallback(args, dataset=dataset))
    trainer.register_callback(CGeneratorTrainingCallback(
        args,
        parameters=model.generator.parameters(),
        criterion=WGANGeneratorLoss(), dataset=dataset))
    trainer.bind_loader('train', train_loader, num_inputs=2)
    # Custom logging configuration so it knows to log our images
    logger = TensorboardLogger(
        log_scalars_every=(1, 'iteration'),
        log_images_every=(args.log_image_frequency, 'iteration'))
    trainer.build_logger(logger, log_directory=args.save_directory)
    logger.observe_state('generated_images')
    logger.observe_state('real_images')
    logger._trainer_states_being_observed_while_training.remove('training_inputs')

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()

    # Generate video from saved images


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')

    # Output directory
    parser.add_argument('--save-directory', type=str,
                        default='output/alderley_patchcwgangp/v2', help='output directory')

    # Configuration
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=250,
                        metavar='N', help='number of epochs')
    parser.add_argument('--image-frequency', type=int, default=60,
                        metavar='N', help='frequency to write images')
    parser.add_argument('--log-image-frequency', type=int, default=100,
                        metavar='N', help='frequency to log images')
    parser.add_argument('--generator-frequency', type=int, default=2,
                        metavar='N', help='frequency to train generator')

    # Hyperparameters
    parser.add_argument('--discriminator-lr', type=float, default=3e-4,
                        metavar='N', help='discriminator learning rate')
    parser.add_argument('--generator-lr', type=float, default=3e-4,
                        metavar='N', help='generator learning rate')
    parser.add_argument('--penalty-weight', type=float, default=20.,
                        metavar='N', help='gradient penalty weight')
    parser.add_argument('--discriminator-instancenorm', type=bool,
                        default=False, metavar='N', help='enable IN')
    parser.add_argument('--generator-instancenorm', type=bool,
                        default=True, metavar='N', help='enable IN')

    # Flags
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--no-ffmpeg', action='store_true',
                        default=True, help='disables video generation')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
