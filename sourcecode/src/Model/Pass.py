import torch
from torch import nn
from math import sqrt

class Pass(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input