import argparse
import sys

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision import transforms

from gan_utils import Reshape, save_args, GANModel, GenerateDataCallback, GeneratorTrainingCallback
from wgan_loss import WGANGeneratorLoss, WGANDiscriminatorLoss


def data_loader(args):
    # Create DataLoader for CIFAR10
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(
        datasets.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    return train_loader


class WGANGeneratorNetwork(nn.Sequential):
    # Network for generation
    # Input is (N, latent_dim)
    def __init__(self, args):
        super(WGANGeneratorNetwork, self).__init__(*[m for m in [
            nn.Linear(args.latent_dim, 1024),
            nn.BatchNorm1d(1024) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Linear(1024, 2 * 2 * 512),
            Reshape(-1, 512, 2, 2),  # N, 512,2,2
            nn.BatchNorm2d(512) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # N, 256,4,4
            nn.BatchNorm2d(256) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # N, 128,8,8
            nn.BatchNorm2d(128) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # N, 64,16,16
            nn.BatchNorm2d(64) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # N, 32,32,32
            nn.BatchNorm2d(32) if args.generator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),  # N, 3,32,32
            nn.Sigmoid()] if m is not None])


class WGANDiscriminatorNetwork(nn.Sequential):
    # Network for discrimination
    # Input is (N, 1, 28, 28)
    def __init__(self, args):
        super(WGANDiscriminatorNetwork, self).__init__(*[m for m in [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # N, 64, 16, 16
            nn.BatchNorm2d(64) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # N, 128, 8, 8
            nn.BatchNorm2d(128) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # N, 128, 4, 4
            nn.BatchNorm2d(256) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # N, 128, 2, 2
            nn.BatchNorm2d(512) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            Reshape(-1, 512 * 2 * 2),  # N, 128*7*7
            nn.Linear(512 * 2 * 2, 1024),  # N, 1024
            nn.BatchNorm1d(1024) if args.discriminator_batchnorm else None,
            nn.LeakyReLU(),
            nn.Linear(1024, 1),  # N, 1
            Reshape(-1)] if m is not None])  # N


def run(args):
    save_args(args)  # save command line to a file for reference
    train_loader = data_loader(args)  # get the data
    model = GANModel(
        args,
        discriminator=WGANDiscriminatorNetwork(args),
        generator=WGANGeneratorNetwork(args))

    # Build trainer
    trainer = Trainer(model)
    trainer.build_criterion(WGANDiscriminatorLoss(penalty_weight=args.penalty_weight, model=model))
    trainer.build_optimizer('Adam', model.discriminator.parameters(), lr=args.discriminator_lr)
    trainer.save_every((1, 'epochs'))
    trainer.save_to_directory(args.save_directory)
    trainer.set_max_num_epochs(args.epochs)
    trainer.register_callback(GenerateDataCallback(args))
    trainer.register_callback(GeneratorTrainingCallback(
        args,
        parameters=model.generator.parameters(),
        criterion=WGANGeneratorLoss()))
    trainer.bind_loader('train', train_loader)
    # Custom logging configuration so it knows to log our images
    logger = TensorboardLogger(
        log_scalars_every=(1, 'iteration'),
        log_images_every=(args.log_image_frequency, 'iteration'))
    trainer.build_logger(logger, log_directory=args.save_directory)
    logger.observe_state('generated_images')
    logger.observe_state('real_images')
    # logger._trainer_states_being_observed_while_training.remove('training_inputs')

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch GAN Example')

    # Output directory
    parser.add_argument('--save-directory', type=str, default='output/cifar10_wgangp/v1', help='output directory')

    # Configuration
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs')
    parser.add_argument('--image-frequency', type=int, default=10, metavar='N', help='frequency to write images')
    parser.add_argument('--log-image-frequency', type=int, default=100, metavar='N', help='frequency to log images')
    parser.add_argument('--generator-frequency', type=int, default=10, metavar='N', help='frequency to train generator')

    # Hyperparameters
    parser.add_argument('--latent-dim', type=int, default=100, metavar='N', help='latent dimension')
    parser.add_argument('--discriminator-lr', type=float, default=3e-4, metavar='N', help='discriminator learning rate')
    parser.add_argument('--generator-lr', type=float, default=3e-4, metavar='N', help='generator learning rate')
    parser.add_argument('--penalty-weight', type=float, default=10., metavar='N', help='gradient penalty weight')
    parser.add_argument('--discriminator-batchnorm', type=bool, default=False, metavar='N', help='enable BN')
    parser.add_argument('--generator-batchnorm', type=bool, default=True, metavar='N', help='enable BN')

    # Flags
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-ffmpeg', action='store_true', default=False, help='disables video generation')

    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run(args)


if __name__ == '__main__':
    main(sys.argv[1:])
