'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            var1 += k
        for i in range(Config.WEIGHT_BIT // Config.DEVICE_BIT):
            j = 2 ** (i * Config.DEVICE_BIT)
            var2 += j
        # Calculate the standard deviation of the noise based on variation and scale
        var = ((pow(var1, 1 / 2) * scale) / var2) * variation
        # Generate noise from a normal distribution and add to the weight tensor
        return torch.normal(mean=0., std=var, size=weight.size()).to(Config.DEVICE)


def conv5x5(in_planes: int, out_planes: int, stride, padding ) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding, bias=False)

def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)

class LeNet(nn.Module):
    def __init__(self,in_channels,is_train,noise_backbone):
        super(LeNet, self).__init__()
        self.conv1 = conv5x5(1, 6, stride=1, padding=2)
        self.conv2 = conv5x5(6, 16, stride=1, padding=0)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.is_train = is_train
        self.noise_backbone = noise_backbone
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

    def forward(self, x):
        if self.is_train:
            out = F.relu(inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=2))
            out = self.pool(out)
            out = F.relu(inf_with_noise(out, self.conv2.weight, self.noise_backbone, stride=1, padding=0))
            out = self.pool(out)
            # print('if',out.shape)
            out = out.view(out.size(0), -1)
            out = F.relu(F.linear(out, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias))
            out = F.relu(F.linear(out, self.fc2.weight + generate_noise(self.fc2.weight, self.noise_backbone), self.fc2.bias))
            out = F.linear(out, self.fc3.weight + generate_noise(self.fc3.weight, self.noise_backbone), self.fc3.bias)
        else:
            out = F.relu(self.conv1(x))
            out = self.pool(out)
            out = F.relu(self.conv2(out))
            out = self.pool(out)
            # print('else:',out.shape)
            out = out.view(out.size(0), -1)
            out = F.relu(self.fc1(out))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
        return out
