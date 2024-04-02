import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import torch

from sram_linear import SRAMLinear
from sram_conv2d import SRAMConv2D

class SRAMWrapper(nn.Module):
    """
    SRAMWrapper class for replacing linear and convolutional layers in a PyTorch model with SRAM-based layers.

    Parameters:
    - sram_linear_model: Instance of SRAMLinear for replacing linear layers
    - sram_conv2d_model: Instance of SRAMConv2D for replacing convolutional layers
    """

    def __init__(self, sram_linear_model, sram_conv2d_model):
        super(SRAMWrapper, self).__init__()
        self.sram_linear_model = sram_linear_model
        self.sram_conv2d_model = sram_conv2d_model

        self._replace_layers()

    def _replace_layers(self):
        """
        Private method to replace linear and convolutional layers in the model with SRAM-based layers.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                sram_linear_layer = SRAMReplacementLinear(self.sram_linear_model)
                setattr(self, name, sram_linear_layer)

            elif isinstance(module, nn.Conv2d):
                sram_conv2d_layer = SRAMReplacementConv2D(self.sram_conv2d_model)
                setattr(self, name, sram_conv2d_layer)

    def forward(self, x):
        """
        Forward pass through the SRAM-wrapped model.

        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor
        """
        return super(SRAMWrapper, self).forward(x)

class SRAMReplacementLinear(nn.Module):
    """
    SRAMReplacementLinear class for replacing linear layers in a PyTorch model with SRAM-based linear layers.

    Parameters:
    - sram_linear_model: Instance of SRAMLinear for SRAM-based calculations
    """

    def __init__(self, sram_linear_model):
        super(SRAMReplacementLinear, self).__init__()
        self.sram_linear_model = sram_linear_model

    def forward(self, x):
        """
        Forward pass through the SRAM-based linear layer.

        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor
        """
        x_np = x.detach().cpu().numpy()
        output_np = self.sram_linear_model.linear(x_np, self.weights, noise_sram=None)
        output_tensor = torch.tensor(output_np, dtype=x.dtype, device=x.device)
        return output_tensor

class SRAMReplacementConv2D(nn.Module):
    """
    SRAMReplacementConv2D class for replacing convolutional layers in a PyTorch model with SRAM-based convolutional layers.

    Parameters:
    - sram_conv2d_model: Instance of SRAMConv2D for SRAM-based calculations
    """

    def __init__(self, sram_conv2d_model):
        super(SRAMReplacementConv2D, self).__init__()
        self.sram_conv2d_model = sram_conv2d_model

    def forward(self, x):
        """
        Forward pass through the SRAM-based convolutional layer.

        Parameters:
        - x: Input tensor

        Returns:
        - Output tensor
        """
        x_np = x.detach().cpu().numpy()
        output_np = self.sram_conv2d_model.conv2d(x_np, self.weights, noise_sram=None)
        output_tensor = torch.tensor(output_np, dtype=x.dtype, device=x.device)
        return output_tensor

if __name__ == "__main__":

    model_name = 'resnet18'  
    pretrained = False

    model = timm.create_model(model_name, pretrained=pretrained)

    print(model)

    sram_wrapper = SRAMWrapper(SRAMLinear, SRAMConv2D)

    # 用 SRAM 模型替代预训练模型中的线性层和卷积层
    sram_wrapper.replace_layers(model)
    print(sram_wrapper)

    # 进行推理
    input_tensor = torch.randn((1, 3, 224, 224))  # 假设输入是 (batch_size, channels, height, width)
    output_original = sram_wrapper(input_tensor)
    output = sram_wrapper(input_tensor)

    # 输出结果
    print("SRAM-wrapped Model Output:", output)
