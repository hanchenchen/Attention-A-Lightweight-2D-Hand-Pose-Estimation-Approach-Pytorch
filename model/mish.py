import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, inputs):
        x = inputs * nn.Tanh()(nn.Softplus()(inputs))
        return x

