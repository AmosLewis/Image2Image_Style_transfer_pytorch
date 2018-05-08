import argparse
import math
import os
import sys
import numpy as np
import functools
from PIL import Image

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F


class UnetUpsample(nn.Module):

    def __init__(self):
        super(UnetUpsample, self).__init__()

        self.initial_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.input_conv_layers = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        ])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_conv_layers = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        ])
        self.output_upconvs = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(256, 128, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(128, 64, kernel_size=2, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),
        ])
        self.final_layer = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

        self.middle_layer = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(0.5)
        self.intermediate_activation = nn.ReLU()

        self.instance_norms = nn.ModuleList([
            nn.InstanceNorm2d(32),
            nn.InstanceNorm2d(64),
            nn.InstanceNorm2d(128),
            nn.InstanceNorm2d(256)
        ])

    def combine_tensors(self, small, big, i):
        small = self.upsample(small)
        small = self.output_upconvs[i](small)

        diffX = big.size()[2] - small.size()[2]
        diffY = big.size()[3] - small.size()[3]
        small = F.pad(small, (diffX // 2, (diffX + 1) // 2,
                        diffY // 2, (diffY + 1) // 2))

        combined = torch.cat([big, small], dim=1)
        return combined

    def forward(self, y):

        current_input = self.initial_layer(y)
        encoder_states = []

        for i, layer in enumerate(self.input_conv_layers):
            current_input = self.dropout(self.intermediate_activation(self.instance_norms[i](layer(current_input))))
            encoder_states.append(current_input)
            current_input = self.pool(current_input)

        current_input = self.middle_layer(current_input)

        for i, layer in enumerate(self.output_conv_layers):
            current_input = self.combine_tensors(current_input, encoder_states[-1 - i], i)
            current_input = self.dropout(self.intermediate_activation(self.instance_norms[-1 - i](layer(current_input))))
        current_input = self.final_activation(self.final_layer(current_input))


        return current_input

