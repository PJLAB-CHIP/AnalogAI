import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import operator as op
import torch



from dataclasses import dataclass
from typing import Type, List

@dataclass
class Config:
    DATA_PATH: str = 'load_dataset/datasets'
    BATCH_SIZE: int = 128
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR: float = 1e-3
    WEIGHT_BIT: int = 8
    DEVICE_BIT: int = 2
    EPOCH: int = 200
    MC_times: int = 200


def generate_noise(weight, variation):
    """
    Adds noise to a given weight tensor.
    Args:
        weight (torch.Tensor): The weight tensor to which noise will be added.
        variation (float): The degree of variation (noise) to be added.
    """
    if variation == 0:
        return torch.zeros_like(weight).to(Config.DEVICE)  # No noise if variation is 0
    else:
        scale = weight.abs().max().item()  # Maximum absolute value of the weight tensor
        var1 = 0
        var2 = 0
        for i in range(Config.WEIGHT_BIT // Config.DEVICE_BIT):
            k = 2 ** (2 * i * Config.DEVICE_BIT)
            var1 += k #近似所有量化级别的总噪声？
        for i in range(Config.WEIGHT_BIT // Config.DEVICE_BIT):
            j = 2 ** (i * Config.DEVICE_BIT)
            var2 += j
        # Calculate the standard deviation of the noise based on variation and scale
        var = ((pow(var1, 1 / 2) * scale) / var2) * variation
        # Generate noise from a normal distribution and add to the weight tensor
        return torch.normal(mean=0., std=var, size=weight.size()).to(Config.DEVICE)

def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)

class MLPS(nn.Module, ):
        def __init__(self):
            super(MLPS, self).__init__()
            self.nb_classes = 10
            self.img_rows = 32
            self.img_cols = 32
            
        
            self.classifier = nn.Sequential(
                nn.Linear(self.img_rows*self.img_cols, 15),
                nn.ReLU(),
                nn.Linear(15, 10),
                nn.ReLU(),
                nn.Linear(10, self.nb_classes))

        def forward(self, x):
            x = x.view(-1, self.img_rows*self.img_cols)
            x = self.classifier(x)
            return F.softmax(x,dim=1)

class MLP(nn.Module, ):
        def __init__(self, in_channels, is_train=False,noise_backbone=0):
            super(MLP, self).__init__()
            self.nb_classes = 10
            self.in_channels = in_channels
            self.img_rows = 28
            self.img_cols = 28
            self.is_train = is_train
            self.noise_backbone = noise_backbone
            self.fc1 = nn.Linear(in_features=self.in_channels*self.img_rows*self.img_cols, out_features=300)
            self.fc2 = nn.Linear(in_features=300, out_features=100)
            self.fc3 = nn.Linear(in_features=100, out_features=self.nb_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(-1, self.img_rows*self.img_cols)
            if self.is_train:
                out = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)
                out = self.relu(out)
                out = F.linear(out, self.fc2.weight + generate_noise(self.fc2.weight, self.noise_backbone), self.fc2.bias)
                out = self.relu(out)
                out = F.linear(out, self.fc3.weight + generate_noise(self.fc3.weight, self.noise_backbone), self.fc3.bias)
            else:
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                out = self.relu(out)
                out = self.fc3(out)
            return F.softmax(out,dim=1)

class MLPL(nn.Module, ):
        def __init__(self):
            super(MLPL, self).__init__()
            self.nb_classes = 10
            self.img_rows = 32
            self.img_cols = 32

            self.classifier = nn.Sequential(
                nn.Linear(self.img_rows*self.img_cols, 300),
                nn.ReLU(),
                nn.Linear(300, 100),
                nn.ReLU(),
                nn.Linear(100, self.nb_classes))

        def forward(self, x):
            x = x.view(-1, self.img_rows*self.img_cols)
            x = self.classifier(x)
            return F.softmax(x,dim=1)


