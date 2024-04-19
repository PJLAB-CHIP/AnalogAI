import torch
import torch.nn as nn
import torch.nn.functional as F
from paper.AnalogAI.qat.qat import Linear_Q, Conv2d_Q
from qmodule_sram import SramConv2d, SramLinear

class BasicBlockQ(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockQ, self).__init__()
        self.conv1 = Conv2d_Q(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, first_layer=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_Q(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_Q(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class BasicBlockQsram(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockQsram, self).__init__()
        self.conv1 = SramConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, a_bits=8, w_bits=8, backend='SRAM',
            parallelism=64,
            error=4,)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SramConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, a_bits=8, w_bits=8, backend='SRAM',
            parallelism=64,
            error=4,)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SramConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, a_bits=8, w_bits=8, backend='SRAM',
            parallelism=64,
            error=4,),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleneckQ, self).__init__()
        self.conv1 = Conv2d_Q(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d_Q(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d_Q(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2d_Q(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetQ(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetQ, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d_Q(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear_Q(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class ResNetQsram(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetQsram, self).__init__()
        self.in_planes = 64

        self.conv1 = SramConv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False, a_bits=8, w_bits=8, backend='SRAM',
            parallelism=64,
            error=4,)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = SramLinear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18Q():
    return ResNetQ(BasicBlockQ, [2, 2, 2, 2])

def ResNet18Qsram():
    return ResNetQsram(BasicBlockQsram, [2, 2, 2, 2])


def ResNet34Q():
    return ResNetQ(BasicBlockQ, [3, 4, 6, 3])


def ResNet50Q():
    return ResNetQ(BottleneckQ, [3, 4, 6, 3])


def ResNet101Q():
    return ResNetQ(BottleneckQ, [3, 4, 23, 3])


def ResNet152Q():
    return ResNetQ(BottleneckQ, [3, 8, 36, 3])