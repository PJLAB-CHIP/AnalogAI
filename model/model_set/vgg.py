'''VGG11/13/16/19 in Pytorch.'''
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


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
def negative_feedback_noise(data, weight, noise_local, noise_feedback, stride, padding):
    o_i = F.conv2d(data, weight + generate_noise(weight, noise_local), stride=stride, padding=padding)
    o_f = F.conv2d(data, weight + generate_noise(weight, noise_feedback), stride=stride, padding=padding)
    return o_i-o_f
    
class VGGReturnFeature(nn.Module):
    def __init__(self, 
                 vgg_name, 
                 in_channels):
        super(VGGReturnFeature, self).__init__()
        self.in_channels = in_channels
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x, noise_intensity=0):
        feature_maps = []
        out = x
        for idx,layer in enumerate(self.features):
            # 给Conv2d添加特定强度噪声
            if isinstance(layer,nn.Conv2d):
                """version1"""
                # noise = generate_noise(self.features[idx].weight,noise_intensity)
                noise = torch.randn_like(layer.weight) * noise_intensity
                self.features[idx].weight = nn.Parameter(self.features[idx].weight + noise)

                # noise = generate_noise(layer.weight,noise_intensity[idx])
                # # noise = torch.randn_like(layer.weight) * noise_intensity
                # self.features[idx].weight = nn.Parameter(self.features[idx].weight + noise)
                """version2"""
                # self.features[idx] = inf_with_noise(out,self.features[idx].weight,
                #                                     noise_intensity[idx],
                #                                     self.features[idx].stride,
                #                                     self.features[idx].padding)
            out = layer(out)
            if isinstance(layer,nn.Conv2d):
                feature_maps.append(out) # TODO: 仅在Conv2d后将feature添加到列表中
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, feature_maps

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(self.in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                self.in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def inf_with_noise(data, weight, noise, stride, padding):
    return F.conv2d(data, weight + generate_noise(weight, noise), stride=stride, padding=padding)


class VGG8(nn.Module):
    def __init__(self, in_channels, num_classes, noise_backbone):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.noise_backbone = noise_backbone

        self.conv1 = conv3x3(self.in_channels, 64, stride=1, padding=1)
        self.conv2 = conv3x3(64, 128, stride=1, padding=1)
        self.conv3 = conv3x3(128, 256, stride=1, padding=1)
        self.conv4 = conv3x3(256, 256, stride=1, padding=1)
        self.conv5 = conv3x3(256, 512, stride=1, padding=1)
        self.conv6 = conv3x3(512, 512, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=8192, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.num_classes)
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

        x = inf_with_noise(x, self.conv1.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn1(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv2.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        x = inf_with_noise(x, self.conv3.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn3(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv4.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.pool(x)

        x = inf_with_noise(x, self.conv5.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn5(x)
        x = self.relu(x)

        x = inf_with_noise(x, self.conv6.weight, self.noise_backbone, stride=1, padding=1)
        x = self.bn6(x)
        x = self.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.linear(x, self.fc1.weight + generate_noise(self.fc1.weight, self.noise_backbone), self.fc1.bias)
        x = self.relu(x)
        output = F.linear(x, self.fc2.weight + generate_noise(self.fc2.weight, self.noise_backbone), self.fc2.bias)

        return output

def vgg8(in_channels, num_classes, noise_backbone):
    return VGG8(in_channels, num_classes, noise_backbone)

if __name__ == '__main__':
    vgg = VGG('VGG11',in_channels=1)
    print(vgg)