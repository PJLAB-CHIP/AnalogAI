# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function


##  Activation unsigned 8bit
# ********************* range_trackers *********************
class RangeTracker(nn.Module):
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.q_level == 'L':    # A,min_max_shape=(1, 1, 1, 1),layer level
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min_max_shape=(N, 1, 1, 1),channel level
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]

        self.update_range(min_val, max_val)

class GlobalRangeTracker(RangeTracker):  # W,min_max_shape=(N, 1, 1, 1),channel级,取本次和之前相比的min_max —— (N, C, W, H)
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))

class AveragedRangeTracker(RangeTracker):  # A,min_max_shape=(1, 1, 1, 1),layer级,取running_min_max —— (N, C, W, H)
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        if self.first_a == 0:
            self.first_a.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)

# ********************* quantizers*********************
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)      # quantize scale factor
        self.register_buffer('zero_point', None) # quantize zero-point

    def update_params(self):
        raise NotImplementedError

    # quantize
    def quantize(self, input):
        output = input * self.scale - self.zero_point
        return output

    def round(self, input):
        output = Round.apply(input)
        return output

    # clamp
    def clamp(self, input):
        output = torch.clamp(input, self.min_val, self.max_val)
        return output

    # dequantize
    def dequantize(self, input):
        output = (input + self.zero_point) / self.scale
        return output

    def forward(self, input):
        if self.bits == 32:
            output = input
        elif self.bits == 1:
            #print('！Binary quantization is not supported ！')
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.update_params()
            output = self.quantize(input)    # quantize
            output = self.round(output)
            output = self.clamp(output)      # clamp
            output = self.dequantize(output) # dequantize
        return output

class SignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(-(1 << (self.bits - 1))))
        self.register_buffer('max_val', torch.tensor((1 << (self.bits - 1)) - 1))

class UnsignedQuantizer(Quantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('min_val', torch.tensor(0))
        self.register_buffer('max_val', torch.tensor((1 << self.bits) - 1))

# symmetrical quantization
class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))  
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val)) 
        self.scale = quantized_range / float_range      
        self.zero_point = torch.zeros_like(self.scale)  

# Asymmetrical quantization
class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        quantized_range = self.max_val - self.min_val  
        float_range = self.range_tracker.max_val - self.range_tracker.min_val   
        self.scale = quantized_range / float_range  
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)  
        # print("self.zero_point: ",self.zero_point)


# W binary weight
class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):

        grad_input = grad_output.clone()

        return grad_input
    
def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean) 
    w.data.clamp(-1.0, 1.0) 
    return w

class weight_bin(nn.Module):
  def __init__(self, ):
    super().__init__()

  def binary(self, input):
    output = Binary_w.apply(input)
    return output

  def forward(self, input):

    output = meancenter_clampConvParams(input) 

    E = torch.mean(torch.abs(output), (-2, -1), keepdim=True)

    alpha = E

    output = self.binary(output)

    output = output * alpha 

    return output


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
        # Instantiating the A and W Quantizers
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = weight_bin()
        self.first_layer = first_layer

    def forward(self, input):
        # quantize A and W
        if not self.first_layer:
            input = self.activation_quantizer(input)

        q8_input = input
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
        # Instantiating the A and W Quantizers
        self.activation_quantizer = AsymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = weight_bin()
        self.first_layer = first_layer

    def forward(self, input):
        # quantize A and W

        if not self.first_layer:
            input = self.activation_quantizer(input)
        q8_input = input
        bin_weight = self.weight_quantizer(self.weight)

        output = F.linear(
            input = q8_input,
            weight = bin_weight,
            bias=self.bias)
        return output
