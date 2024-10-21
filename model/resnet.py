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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)

class BasicBlock(nn.Module):
    def __init__(
            self,
            noise_backbone,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            is_train: bool = False
    ):
        super().__init__()
        self.is_train = is_train
        self.noise_backbone = noise_backbone
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.conv1 = conv3x3(self.in_planes, self.out_planes, stride=self.stride, padding=1)
        self.conv2 = conv3x3(self.out_planes, self.out_planes, stride=1, padding=1)
        self.unit_conv = conv1x1(self.in_planes, self.out_planes, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(self.out_planes)
        self.bn2 = nn.BatchNorm2d(self.out_planes)
        self.bn3 = nn.BatchNorm2d(self.out_planes)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        if self.is_train:
            identity = x
            out = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=self.stride, padding=1)
            out = self.bn1(out)
            out = self.relu(out)
            out = inf_with_noise(out, self.conv2.weight, self.noise_backbone, stride=1, padding=1)
            out = self.bn2(out)
            if self.in_planes != self.out_planes or self.stride != 1:
                    identity = self.bn3(inf_with_noise(x, self.unit_conv.weight, self.noise_backbone, stride=self.stride, padding=0))
            out += identity
            out = self.relu(out)
        else:
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.in_planes != self.out_planes or self.stride != 1:
                    identity = self.bn3(self.unit_conv(x))
            out += identity
            out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(
            self,
            in_channels,
            num_classes,
            block: Type[BasicBlock],
            layers: List[int],
            noise_backbone=None,
            is_train=False
            # noise_feedback

    ):
        super().__init__()
        self.is_train = is_train
        self.noise_backbone = noise_backbone  # noise var for backbone
        self.conv1 = conv3x3(in_channels, 64, stride=1, padding=1)  # first conv layer
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=512, out_features=num_classes)
        self.init_weights()
        # self.noise_feedback = [0.1,0.2,0.3]
        # self.cal_times = 0

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(
            self,
            block: Type[BasicBlock],
            in_planes: int,
            out_planes: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        layers = []
        layers.append(block(self.noise_backbone, in_planes, out_planes, stride=stride))
        for _ in range(1, blocks):
            layers.append(block(self.noise_backbone, out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print('x.shape:', x.shape)
        # print('@@@--->',self.conv1,self.conv1.weight)
        if self.is_train:
            x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.is_train:
            x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)
        else:
            x = self.fc1(x)
        return x


def resnet18(in_channels,noise_backbone=0, is_train=False):
    return ResNet(in_channels, 10, BasicBlock, [2, 2, 2, 2], noise_backbone, is_train)

def ResNet34(in_channels):
    return ResNet(BasicBlock, [3, 4, 6, 3],in_channels=in_channels)




