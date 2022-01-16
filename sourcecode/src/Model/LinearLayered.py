import torch
from torch import nn
from math import sqrt

class LinearLayered(nn.Module):
    def __init__(self, in_features, out_features, numOfLayers):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.numOfLayers  = numOfLayers
        self.weights = []
        self.biases = []

        # 重みを格納する行列の定義
        for i in range(numOfLayers):
            if(i==0):
                numOfInput =  self.in_features
            if(0<i):
                numOfInput =  1 / out_features
            k = 1 / numOfInput
            weight = torch.empty(out_features, numOfInput).uniform_(-sqrt(k), sqrt(k))
            self.weights.append(nn.Parameter(weight))

        # バイアスを格納するベクトルの定義
        for i in range(numOfLayers):
            bias = torch.empty(out_features).uniform_(-k, k)
            self.biases.append(nn.Parameter(bias))

    def forward(self, input):
        for i in range(self.numOfLayers):
            input = torch.nn.functional.linear(input, self.weights[i], self.bias[i])
        return input

