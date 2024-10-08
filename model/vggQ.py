'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from paper.AnalogAI.qat.qat import Linear_Q, Conv2d_Q

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGQ(nn.Module):
    def __init__(self, vgg_name):
        super(VGGQ, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = Linear_Q(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        first = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if(first):
                    layers += [Conv2d_Q(in_channels, x, kernel_size=3, padding=1, first_layer=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                    first = 0
                else:
                    layers += [Conv2d_Q(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGGQ('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
