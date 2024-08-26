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

def negative_feedback_noise(data, weight, noise_local, noise_feedback, stride, padding):
    o_i = F.conv2d(data, weight + generate_noise(weight, noise_local), stride=stride, padding=padding)
    o_f = F.conv2d(data, weight + generate_noise(weight, noise_feedback), stride=stride, padding=padding)
    return o_i-o_f

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

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

# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks,in_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

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


# def ResNet18(in_channels):
#     return ResNet(BasicBlock, [2, 2, 2, 2],in_channels=in_channels)

# def resnet18(in_channels,noise_backbone,noise_feedback):
#     return ResNet(in_channels, 10, BasicBlock, [2, 2, 2, 2], noise_backbone, noise_feedback)
def resnet18(in_channels,noise_backbone=None):
    return ResNet(in_channels, 10, BasicBlock, [2, 2, 2, 2], noise_backbone)


def ResNet34(in_channels):
    return ResNet(BasicBlock, [3, 4, 6, 3],in_channels=in_channels)


# def ResNet50(in_channels):
#     return ResNet(Bottleneck, [3, 4, 6, 3],in_channels=in_channels)


# def ResNet101(in_channels):
#     return ResNet(Bottleneck, [3, 4, 23, 3],in_channels=in_channels)


# def ResNet152(in_channels):
#     return ResNet(Bottleneck, [3, 8, 36, 3],in_channels=in_channels)


import numpy as np

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

def cosine_schedule(T, beta_min=0.5, beta_max=1.5, s=0.008):
    def f(t, T, s):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T, s)
    
    for t in range(T + 1):
        alphas.append(f(t, T, s) / f0)
    
    betas = []
    
    for t in range(1, T + 1):
        beta = min(1 - alphas[t] / alphas[t - 1], 0.999)
        betas.append(beta)
    betas.reverse()
    
    betas = np.array(betas)
    betas = beta_min + (betas - betas.min()) * (beta_max - beta_min) / (betas.max() - betas.min())
    
    return betas

def counter_layer(model):
    layer_counter = 0
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            layer_counter += 1
    return layer_counter

def assign_coefficients_to_layers(model, min_value=0.5, max_value=1.5):
    coefficients = {}
    layer_counter = counter_layer(model)
    coefficients_schedule = cosine_schedule(layer_counter, min_value, max_value)

    layer_idx = 0
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            coefficient = coefficients_schedule[layer_idx]
            coefficients[name] = coefficient
            layer_idx += 1

    return coefficients

class NegativeResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block: Type['NegativeBasicBlock'],
        layers: List[int],
        noise_backbone: float = 0.1,
        noise_feedback: float = 0.1,
        is_train: bool = True,
        is_negative: bool = True,
        schedule_type: str = 'cosine',
        beta_min: float = 0.05,
        beta_max: float = 0.2
    ):
        super(NegativeResNet, self).__init__()
        self.is_train = is_train
        self.is_negative = is_negative
        self.noise_backbone = noise_backbone
        self.noise_feedback_base = noise_feedback
        
        # 定义初始的卷积层和批归一化层
        self.conv1 = conv3x3(in_channels, 64, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 定义残差层
        self.layer1 = self._make_layer(block, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        
        # 定义平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # 初始化权重
        self._initialize_weights()
        
        # 分配反馈系数
        self.coefficients = assign_coefficients_to_layers(
            self,
            schedule_type=schedule_type,
            beta_min=beta_min,
            beta_max=beta_max
        )

    def _make_layer(self, block, in_planes, out_planes, blocks, stride=1):
        layers = []
        layers.append(
            block(
                in_planes,
                out_planes,
                stride,
                self.is_train,
                self.is_negative,
                self.noise_backbone,
                self.noise_feedback_base,
                self.coefficients
            )
        )
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_planes,
                    out_planes,
                    1,
                    self.is_train,
                    self.is_negative,
                    self.noise_backbone,
                    self.noise_feedback_base,
                    self.coefficients
                )
            )
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.is_train and self.is_negative:
            coeff = self.coefficients['conv1']
            x = negative_feedback_noise(
                x,
                self.conv1.weight,
                self.noise_backbone,
                self.noise_feedback_base * coeff,
                stride=1,
                padding=1
            )
        elif self.is_train:
            x = inf_with_noise(
                x,
                self.conv1.weight,
                self.noise_backbone,
                stride=1,
                padding=1
            )
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
        
        if self.is_train and self.is_negative:
            coeff = self.coefficients['fc']
            weight = self.fc.weight
            bias = self.fc.bias
            x = F.linear(
                x,
                weight + generate_noise(weight, self.noise_backbone) - generate_noise(weight, self.noise_feedback_base * coeff),
                bias
            )
        elif self.is_train:
            weight = self.fc.weight
            bias = self.fc.bias
            x = F.linear(
                x,
                weight + generate_noise(weight, self.noise_backbone),
                bias
            )
        else:
            x = self.fc(x)
        
        return x

class NegativeBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int = 1,
        is_train: bool = True,
        is_negative: bool = True,
        noise_backbone: float = 0.1,
        noise_feedback_base: float = 0.1,
        coefficients: dict = None
    ):
        super(NegativeBasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.is_train = is_train
        self.is_negative = is_negative
        self.noise_backbone = noise_backbone
        self.noise_feedback_base = noise_feedback_base
        self.coefficients = coefficients
        
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
        identity = x
        
        # 第一个卷积层
        if self.is_train and self.is_negative:
            coeff = self.coefficients[self.conv1_name()]
            out = negative_feedback_noise(
                x,
                self.conv1.weight,
                self.noise_backbone,
                self.noise_feedback_base * coeff,
                stride=self.conv1.stride,
                padding=self.conv1.padding
            )
        elif self.is_train:
            out = inf_with_noise(
                x,
                self.conv1.weight,
                self.noise_backbone,
                stride=self.conv1.stride,
                padding=self.conv1.padding
            )
        else:
            out = self.conv1(x)
        
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积层
        if self.is_train and self.is_negative:
            coeff = self.coefficients[self.conv2_name()]
            out = negative_feedback_noise(
                out,
                self.conv2.weight,
                self.noise_backbone,
                self.noise_feedback_base * coeff,
                stride=self.conv2.stride,
                padding=self.conv2.padding
            )
        elif self.is_train:
            out = inf_with_noise(
                out,
                self.conv2.weight,
                self.noise_backbone,
                stride=self.conv2.stride,
                padding=self.conv2.padding
            )
        else:
            out = self.conv2(out)
        
        out = self.bn2(out)
        if self.in_planes != self.out_planes or self.stride != 1:
            if self.is_train and self.is_negative:
                coeff = self.coefficients[self.unit_conv_name()]
                identity = self.bn3(negative_feedback_noise(
                    x,
                    self.unit_conv.weight,
                    self.noise_backbone,
                    self.noise_feedback_base * coeff,
                    stride=self.stride, padding=0
                ))

            elif self.is_train:
                identity = self.bn3(inf_with_noise(x, self.unit_conv.weight, self.noise_backbone, stride=self.stride, padding=0))

            else:
                identity = self.bn3(self.unit_conv(x))
        out += identity
        out = self.relu(out)
        
        return out

    def conv1_name(self):
        return f'{self._get_name()}.conv1'

    def conv2_name(self):
        return f'{self._get_name()}.conv2'

    def unit_conv_name(self):
        return f'{self._get_name()}.unit_conv'


def negative_resnet18(num_classes=1000, is_train=True, is_negative=True):
    return NegativeResNet(
        in_channels=3,
        num_classes=num_classes,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        noise_backbone=0.1,
        noise_feedback=0.1,
        is_train=is_train,
        is_negative=is_negative,
        schedule_type='cosine',
        beta_min=0.05,
        beta_max=0.2
    )

if __name__ == '__main__':
    model = negative_resnet18(num_classes=10, is_train=True, is_negative=True)
    input_tensor = torch.randn(8, 3, 32, 32)
    output = model(input_tensor)
    print(output)