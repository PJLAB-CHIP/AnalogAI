import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block of a residual network with option for the skip connection."""

    def __init__(self, in_ch, hidden_ch, use_conv=False, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_ch)

        if use_conv:
            self.convskip = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, stride=stride)
        else:
            self.convskip = None

    def forward(self, x):
        """Forward pass"""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.convskip:
            x = self.convskip(x)
        y += x
        return F.relu(y)


def concatenate_layer_blocks(in_ch, hidden_ch, num_layer, first_layer=False):
    """Concatenate multiple residual block to form a layer.

    Returns:
       List: list of layer blocks
    """
    layers = []
    for i in range(num_layer):
        if i == 0 and not first_layer:
            layers.append(ResidualBlock(in_ch, hidden_ch, use_conv=True, stride=2))
        else:
            layers.append(ResidualBlock(hidden_ch, hidden_ch))
    return layers


def create_resnet32_model(N_CLASSES):
    """ResNet34 inspired analog model.

    Returns:
       nn.Modules: created model
    """

    block_per_layers = (3, 4, 6, 3)
    base_channel = 16
    channel = (base_channel, 2 * base_channel, 4 * base_channel)

    l0 = nn.Sequential(
        nn.Conv2d(3, channel[0], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
    )

    l1 = nn.Sequential(
        *concatenate_layer_blocks(channel[0], channel[0], block_per_layers[0], first_layer=True)
    )
    l2 = nn.Sequential(*concatenate_layer_blocks(channel[0], channel[1], block_per_layers[1]))
    l3 = nn.Sequential(*concatenate_layer_blocks(channel[1], channel[2], block_per_layers[2]))
    l4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(channel[2], N_CLASSES))

    return nn.Sequential(l0, l1, l2, l3, l4)

# model = create_model(10)
# print(model)
#----------------------------------------------
