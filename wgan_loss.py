import argparse
import sys

import torch
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad

class WGANDiscriminatorLoss(nn.Module):
    def __init__(self, penalty_weight, model):
        super(WGANDiscriminatorLoss, self).__init__()
        self.model = model
        self.penalty_weight = penalty_weight

    # Run discriminator
    def discriminate(self, xmix):
        return self.model.discriminate(xmix)

    # Loss function for discriminator
    def forward(self, inp, _):
        # Targets are ignored
        yreal, yfake = inp  # unpack inputs

        # Main loss calculation
        wgan_loss = yfake.mean() - yreal.mean()

        # Gradient penalty
        xreal = self.model._state_hooks['xreal']
        xfake = self.model._state_hooks['xfake']
        # Random linear combination of xreal and xfake
        alpha = Variable(torch.rand(xreal.size(0), 1, 1, 1, out=xreal.data.new()))
        xmix = (alpha * xreal) + ((1. - alpha) * xfake)
        # Run discriminator on the combination
        ymix = self.discriminate(xmix)
        # Calculate gradient of output w.r.t. input
        ysum = ymix.sum()
        grads = grad(ysum, [xmix], create_graph=True)[0]
        gradnorm = torch.sqrt((grads * grads).sum(3).sum(2).sum(1))
        graddiff = gradnorm - 1
        gradpenalty = (graddiff * graddiff).mean() * self.penalty_weight

        # Total loss
        loss = wgan_loss + gradpenalty
        return loss


class WGANGeneratorLoss(nn.BCEWithLogitsLoss):
    # Loss function for generator
    def forward(self, yfake):
        loss = -yfake.mean()
        return loss

    
class CWGANDiscriminatorLoss(WGANDiscriminatorLoss):
    def discriminate(self, xmix):
        y = self.model._state_hooks['y']
        return self.model.discriminate(xmix, y)
