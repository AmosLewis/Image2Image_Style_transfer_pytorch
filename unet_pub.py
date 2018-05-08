import argparse
import math
import os
import sys
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F


class UnetPub(nn.Module):

    def __init__(self):
        super(UnetPub, self).__init__()

        # Input is (N, 3, 256, 512)
        # No BatchNorm after the first layer
        self.initial_layer = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # N, 64, 128, 256
        self.encoder_activation = nn.LeakyReLU(negative_slope=0.2)

        self.encoder_layers = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # N, 128, 64, 128
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # N, 256, 32, 64
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 16, 32
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 8, 16
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 4, 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 2, 4
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 1, 2
        ])

        self.encoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(128),
            nn.InstanceNorm2d(256),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
        ])

        self.decoder_activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 2, 4
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 4, 8
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 8, 16
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # N, 512, 16, 32
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # N, 256, 32, 64
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # N, 128, 64, 128
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)  # N, 64, 128, 256
        ])

        self.decoder_norms = nn.ModuleList([
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(512),
            nn.InstanceNorm2d(256),
            nn.InstanceNorm2d(128),
            nn.InstanceNorm2d(64),
        ])

        self.final_layer = nn.ConvTranspose2d(
            128, 3, kernel_size=4, stride=2, padding=1)  # N, 3, 256, 512
        self.final_activation = nn.ReLU()

    def forward(self, y):

        encoder_states = []

        current_input = self.encoder_activation(self.initial_layer(y))
        encoder_states.append(current_input)

        for i, layer in enumerate(self.encoder_layers):
            current_input = self.encoder_activation(self.encoder_norms[i](layer(current_input)))
            encoder_states.append(current_input)

        for i, layer in enumerate(self.decoder_layers):
            current_input = self.decoder_activation(
                self.dropout(self.decoder_norms[i](layer(current_input))))
            current_input = torch.cat([current_input, encoder_states[-1 - i - 1]], dim=1)

        current_input = self.final_activation(self.final_layer(current_input))

        return current_input
