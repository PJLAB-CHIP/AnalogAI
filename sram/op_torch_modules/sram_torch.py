# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


class Conv2d_Q(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        a_bits=8,
        w_bits=2,
        first_layer=0,
      ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = weight_bin()
        self.first_layer = first_layer

    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)

        q8_input = input
        # print(q8_input)
        bin_weight = self.weight_quantizer(self.weight)
        output = F.conv2d(
            input = q8_input,
            weight = bin_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return output

class Linear_Q(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        a_bits=8,
        w_bits=2,
        first_layer=0,
      ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = weight_bin()
        self.first_layer = first_layer
    def forward(self, input):
        # 量化A和W
        if not self.first_layer:
            input = self.activation_quantizer(input)
        q8_input = input
        bin_weight = self.weight_quantizer(self.weight)

        # 用量化后的A和W做卷积
        #print('Linear,input',q8_input)
        #print('weight',bin_weight)
        output = F.linear(
            input = q8_input,
            weight = bin_weight,
            bias=self.bias)
        return output