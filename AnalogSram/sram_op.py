# -*- coding: utf-8 -*-
# Created on 2024.3.21
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math
import copy
# from model.vit import MultiHeadSelfAttention


class RangeTracker(nn.Module):
    """
        Initialize the RangeTracker.

        Parameters:
        - q_level: Quantization level ('L' for layer-wise or 'C' for channel-wise).
    """
    def __init__(self, q_level):
        super().__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        """
        Update the range of values. To be implemented in subclasses.  
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        """
        Forward pass to calculate and update the range.

        Parameters:
        - input: Input tensor for which the range needs to be tracked.
        """
        # Implementation depends on q_level (Layer or Channel-wise)
        if self.q_level == 'L':    # A,min_max_shape=(1, 1, 1, 1),layer
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.q_level == 'C':  # W,min_max_shape=(N, 1, 1, 1),channel
            min_val = torch.min(torch.min(torch.min(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]
            max_val = torch.max(torch.max(torch.max(input, 3, keepdim=True)[0], 2, keepdim=True)[0], 1, keepdim=True)[0]

        self.update_range(min_val, max_val)

class GlobalRangeTracker(RangeTracker):  
    """
        Initialize the GlobalRangeTracker.

        Parameters:
        - q_level: Quantization level, expected 'C' for channel-wise.
        - out_channels: Number of output channels.
    """
    
    def __init__(self, q_level, out_channels):
        super().__init__(q_level)
        self.register_buffer('min_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('max_val', torch.zeros(out_channels, 1, 1, 1))
        self.register_buffer('first_w', torch.zeros(1))

    def update_range(self, min_val, max_val):
        """
        Update the global range considering the new min and max values.

        Parameters:
        - min_val: New minimum value.
        - max_val: New maximum value.
        """
        # Update min and max values based on new inputs
        temp_minval = self.min_val
        temp_maxval = self.max_val
        if self.first_w == 0:
            self.first_w.add_(1)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            self.min_val.add_(-temp_minval).add_(torch.min(temp_minval, min_val))
            self.max_val.add_(-temp_maxval).add_(torch.max(temp_maxval, max_val))

class AveragedRangeTracker(RangeTracker):  
    """
        Initialize the AveragedRangeTracker.

        Parameters:
        - q_level: Quantization level, expected 'L' for layer-wise.
        - momentum: Momentum for updating the running average.
    """
    def __init__(self, q_level, momentum=0.1):
        super().__init__(q_level)
        self.momentum = momentum
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.register_buffer('first_a', torch.zeros(1))

    def update_range(self, min_val, max_val):
        """
        Update the range using running average method.

        Parameters:
        - min_val: New minimum value.
        - max_val: New maximum value.
        """
        # Update the running average of min and max values
        if self.first_a == 0:
            self.first_a.add_(1)
            min_val = min_val.to(self.min_val.device)
            max_val = max_val.to(self.max_val.device)
            self.min_val.add_(min_val)
            self.max_val.add_(max_val)
        else:
            min_val = min_val.to(self.min_val.device)
            max_val = max_val.to(self.max_val.device)
            self.min_val.mul_(1 - self.momentum).add_(min_val * self.momentum)
            self.max_val.mul_(1 - self.momentum).add_(max_val * self.momentum)

# Round: Custom function to perform rounding operation in forward and backward pass.
class Round(Function):

    @staticmethod
    def forward(self, input):
        """
        Perform rounding operation in the forward pass.

        Parameters:
        - input: Input tensor to be rounded.

        Returns:
        - Rounded tensor.
        """
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        """
        Backward pass for the rounding operation.

        Parameters:
        - grad_output: Gradient tensor from subsequent layers.

        Returns:
        - Gradient tensor for input.
        """
        grad_input = grad_output.clone()
        return grad_input

class Quantizer(nn.Module):
    def __init__(self, bits, range_tracker):
        """
        Initialize the Quantizer module.

        Parameters:
        - bits: Number of bits for quantization.
        - range_tracker: Instance of RangeTracker for range tracking.
        """
        super().__init__()
        self.bits = bits
        self.range_tracker = range_tracker
        self.register_buffer('scale', None)      # 量化比例因子
        self.register_buffer('zero_point', None) # 量化零点

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
            # 'Binary quantization is not supported ！
            assert self.bits != 1
        else:
            self.range_tracker(input)
            self.scale = self.update_params()
            output = self.quantize(input)   # quantize
            output = self.round(output)
            q_output = self.clamp(output)     # clamp
            # output = self.dequantize(output)#  if qat ：dequantize
        return q_output, self.scale

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

class SymmetricQuantizer(SignedQuantizer):

    def update_params(self):
        quantized_range = torch.min(torch.abs(self.min_val), torch.abs(self.max_val))  
        float_range = torch.max(torch.abs(self.range_tracker.min_val), torch.abs(self.range_tracker.max_val)) 
        self.scale = quantized_range / float_range      
        self.zero_point = torch.zeros_like(self.scale)  
        return self.scale


class AsymmetricQuantizer(UnsignedQuantizer):

    def update_params(self):
        quantized_range = self.max_val - self.min_val  # 量化后范围
        float_range = self.range_tracker.max_val - self.range_tracker.min_val   # 量化前范围
        self.scale = quantized_range / float_range  # 量化比例因子
        self.zero_point = torch.round(self.range_tracker.min_val * self.scale)  # 量化零点
        # print("self.zero_point: ",self.zero_point)


# W binary weight
class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        #*******************ste*********************
        grad_input = grad_output.clone()
        #print(grad_input)
        return grad_input

# ********************* W(模型参数)量化(二值) ***********************
def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean) # W中心化(C方向)
    w.data.clamp(-1.0, 1.0) # W截断
    return w

class weight_bin(nn.Module):
  def __init__(self, ):
    super().__init__()

  def binary(self, input):
    output = Binary_w.apply(input)
    return output

  def forward(self, input):
    # **************************************** W二值 *****************************************
    output = meancenter_clampConvParams(input) # W中心化+截断
    # **************** channel级 - E(|W|) ****************
    ##print('output',output)
    #print('abs',torch.abs(output))
    E = torch.mean(torch.abs(output), (-2, -1), keepdim=True)
    #print('mean',E)
    # **************** α(缩放因子) ****************
    alpha = E
    # ************** W —— +-1 **************
    output = self.binary(output)
    # ************** W * α **************
    output = output * alpha # 若不需要α(缩放因子)，注释掉即可

    return output
  
# Comput the MACs of each conv's matrix multiply
class Round(Function):

    @staticmethod
    def forward(self, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input
    
class AnalogSram2d(Function):
    '''
    SRAM Simulation Computing Platform for Error Modeling
    '''
    @staticmethod
    def forward(ctx, output, K, C_in, C_out, groups=1, parallelism=64, error_range=0.01) -> torch.Any:
        col = K * C_in * C_out // groups
        errors_per_output_element = math.ceil(col / parallelism)
        
        # Generate all random errors at once
        total_error_shape = (errors_per_output_element,) + output.shape
        # total_errors = torch.randint(-error_range, error_range + 1, total_error_shape, device=output.device).float()
        total_errors = torch.normal(0, error_range*output, total_error_shape, device=output.device)

        # Round the errors to nearest integers
        total_errors = torch.round(total_errors)
        # Sum up all errors and reshape to match the output shape
        total_errors = total_errors.sum(dim=0)

        output += total_errors
        total_errors=0

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None, None, None, None, None
    

# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class SramConv2d(nn.Conv2d):
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
        w_bits=8,
        backend='SRAM',
        parallelism=128,
        error=4,
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
        # A & W quantizer
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.parallelism = parallelism
        self.error = error
        if backend is not None:
            assert backend == 'SRAM'
            # self.SRAM = AnalogSram(m=parallelism, error=error)
    def forward(self, input):
        # 量化A和W
        input, input_scale = self.activation_quantizer(input)
        w_input, w_scale = self.weight_quantizer(self.weight)
        B, C_in, H, W = input.shape
        C_out, _, K, K = w_input.shape

        output = F.conv2d(
            input = input,
            weight = w_input,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)

        # output = self.SRAM.calculate_mac_operations(output, K, C_in, C_out, self.groups)       
        output = AnalogSram2d.apply(output, K, C_in, C_out, self.groups, self.parallelism, self.error) 
        # dequantize
        output = output / (input_scale*w_scale)

        return output
    
class AnalogSramLinear(Function):
    '''
    SRAM Simulation Computing Platform for Error Modeling
    '''
    @staticmethod
    def forward(ctx, output, w_input, parallelism, error_range) -> torch.Any:
        """
        Simulate SRAM-based 2D operations with error introduction.

        Parameters:
        - output: Output tensor from the convolution operation.
        - K: Kernel size.
        - C_in: Number of input channels.
        - C_out: Number of output channels.
        - groups: Number of groups in convolution.
        - parallelism: Degree of parallelism in operations.
        - error_range: Range for random error introduction.

        Returns:
        - Output tensor with introduced random errors.
        """
        # Generate and sum up all random errors, then add to the output
        weight_rows = w_input.shape[0]
        errors_per_output_element = math.ceil(weight_rows / parallelism)
        
        # Generate all random errors at once
        total_error_shape = (errors_per_output_element,) + output.shape
        # total_errors = torch.randint(-error_range, error_range + 1, total_error_shape, device=output.device).float()
        total_errors = torch.normal(0, error_range*output, total_error_shape, device=output.device)

        # Round the errors to nearest integers
        total_errors = torch.round(total_errors)
        # Sum up all errors and reshape to match the output shape
        total_errors = total_errors.sum(dim=0)

        output += total_errors
        total_errors=0

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the SRAM simulation.

        Parameters:
        - grad_output: Gradient tensor from subsequent layers.

        Returns:
        - Gradient tensor for output, with additional None placeholders.
        """
        grad_input = grad_output.clone()
        return grad_input, None, None, None

class SramLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        a_bits=8,
        w_bits=8,
        backend='SRAM',
        parallelism=128,
        error=4,
      ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias
        )

        # A & W quantizer
        self.activation_quantizer = SymmetricQuantizer(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = SymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.parallelism = parallelism
        self.error = error
        if backend is not None:
            assert backend == 'SRAM'

    def forward(self, input):

        input, input_scale = self.activation_quantizer(input)
        w_input, w_scale = self.weight_quantizer(self.weight)

        output = F.linear(
            input = input,
            weight = w_input,
            bias=self.bias)
        
        output = AnalogSramLinear.apply(output, w_input, self.parallelism, self.error) 
        
        output = output / (input_scale*w_scale)

        return output
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super().__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sram_error_simulator=None, parallelism=128,
                        error=4,):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        # 使用SRAM误差模拟的einsum
        if sram_error_simulator is not None:
            score = F.softmax(sram_einsum("bhif, bhjf->bhij", q, k, 
                                          parallelism=parallelism,
                        error=error,)/self.sqrt_d, dim=-1)
            attn = sram_einsum("bhij, bhjf->bihf", score, v, 
                               parallelism=parallelism,
                        error=error,)
        else:
            score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1)
            attn = torch.einsum("bhij, bhjf->bihf", score, v)

        o = self.dropout(self.o(attn.flatten(2)))
        return o

'''
class SramMultiHeadSelfAttention(MultiHeadSelfAttention):
    def __init__(
        self,
        feats,
        head,
        dropout,
        a_bits=8,
        w_bits=8,
        backend='SRAM',
        parallelism=128,
        error=4,
      ):
        super().__init__(
            feats=feats, 
            head=head,
            dropout=dropout
        )

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        # 使用自定义的SRAMMatrixOperation进行矩阵乘法
        score = SRAMMatrixOperation.apply(torch.einsum("bhif, bhjf->bhij", q, k), self.parallelism, self.error_range)
        score = F.softmax(score / self.sqrt_d, dim=-1)
        attn = SRAMMatrixOperation.apply(torch.einsum("bhij, bhjf->bihf", score, v), self.parallelism, self.error_range)

        o = self.dropout(self.o(attn.flatten(2)))
        return o
'''

def parse_einsum_equation_for_matrix_mul(equation, operands):
    input_eq, _ = equation.split('->')
    input_ops = input_eq.split(',')

    # 找出共有维度
    common_dims = set(input_ops[0]).intersection(input_ops[1])

    # 找出参与乘法的维度，通常是两个操作数中的最后一个共有维度
    common_dim = None
    for dim in reversed(input_ops[0]):
        if dim in common_dims:
            common_dim = dim
            break

    if common_dim is None:
        raise ValueError("No common dimension found for matrix multiplication.")

    # 获取该共有维度在第一个操作数中的索引
    common_dim_idx = input_ops[0].find(common_dim)

    # 获取共有维度的大小
    mac_dims = operands[0].shape[common_dim_idx]

    return mac_dims


def apply_sram_error_to_attention(module, sram_error_simulator, parallelism,
                        error,):

    original_forward = module.forward

    def forward_with_sram_error(x):
        return original_forward(x, sram_error_simulator=sram_error_simulator, parallelism=parallelism,
                        error=error,)
    
    module.forward = forward_with_sram_error


class SRAMErrorSimulator(Function):

    @staticmethod
    def forward(ctx: torch.Any, output, mac_dims, parallelism,
                        error_range,) -> torch.Any:
        errors_per_output_element = math.ceil(mac_dims / parallelism)
        # Generate all random errors at once
        total_error_shape = (errors_per_output_element,) + output.shape
        # total_errors = torch.randint(-error_range, error_range + 1, total_error_shape, device=output.device).float()
        total_errors = torch.normal(0, error_range, total_error_shape, device=output.device)

        # Round the errors to nearest integers
        total_errors = torch.round(total_errors)
        # Sum up all errors and reshape to match the output shape
        total_errors = total_errors.sum(dim=0)

        output += total_errors
        total_errors=0

        return output

    @staticmethod
    def backward(ctx: torch.Any, grad_outputs: torch.Any) -> torch.Any:
        """
        Backward pass for the SRAM simulation.

        Parameters:
        - grad_output: Gradient tensor from subsequent layers.

        Returns:
        - Gradient tensor for output, with additional None placeholders.
        """
        grad_input = grad_outputs.clone()
        return grad_input, None, None, None
    
def calculate_scale_factor(tensor):
    max_val = torch.max(torch.abs(tensor))
    scale_factor = max_val / 127.0
    return scale_factor

def quantize(tensor, scale_factor):
    return tensor / scale_factor

def round(input):
        output = Round.apply(input)
        return output

def clamp(input):
        output = torch.clamp(input, -128, 127)
        return output

def dequantize(quantized_tensor, scale_factor):
    return quantized_tensor * scale_factor



def sram_einsum(equation, *operands, parallelism, error):
    # quantize operands and get scale_factors, respectively
    quantized_operands = []
    scale_factors = []
    for operand in operands:
        scale_factor = calculate_scale_factor(operand)
        quantized_operand = quantize(operand, scale_factor)
        quantized_operand = round(quantized_operand)
        quantized_operand = clamp(quantized_operand)
        quantized_operands.append(quantized_operand)
        scale_factors.append(scale_factor)

    # einsum result
    output = torch.einsum(equation, *quantized_operands)

    # get MAC times
    mac_dims = parse_einsum_equation_for_matrix_mul(equation, operands)

    # apply SRAM error
    output_with_error = SRAMErrorSimulator.apply(output, mac_dims, parallelism, error)

    # dequantize
    combined_scale_factor = math.prod(scale_factors)
    dequantized_output = dequantize(output_with_error, combined_scale_factor)

    return dequantized_output

def convet_sram_op(module, layer_counter, device, a_bits=8, w_bits=8, backend='SRAM', parallelism=128,
        error=4,):
    for name, child in module.named_children():      
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    sram_conv = SramConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        # padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        backend=backend, 
                        parallelism=parallelism,
                        error=error,
                    )
                    sram_conv.bias.data = child.bias
                else:
                    sram_conv = SramConv2d(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        # padding_mode=child.padding_mode,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        backend=backend, 
                        parallelism=parallelism,
                        error=error,
                    )
            
                sram_conv.weight.data = child.weight
                sram_conv.to(device)
                module._modules[name] = sram_conv

        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    sram_linear = SramLinear(
                        child.in_features,
                        child.out_features,
                        bias=True,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        backend=backend, 
                        parallelism=parallelism,
                        error=error,
                    )
                    sram_linear.bias.data = child.bias
                else:
                    sram_linear = SramLinear(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        a_bits=a_bits,
                        w_bits=w_bits,
                        backend=backend, 
                        parallelism=parallelism,
                        error=error,
                    )
                sram_linear.weight.data = child.weight
                sram_linear.to(device)
                module._modules[name] = sram_linear

        elif isinstance(child, MultiHeadSelfAttention):
            # 直接递归处理MultiHeadSelfAttention内部的层
            convet_sram_op(
                child, 
                layer_counter, 
                device, 
                a_bits, 
                w_bits, 
                backend, 
                parallelism, 
                error)
            # sram_MHA = SramMultiHeadSelfAttention(
            #     child.feats,
            #     child.head,
            #     child.dropout,
            #     a_bits=a_bits,
            #     w_bits=w_bits,
            #     backend=backend, 
            #     parallelism=parallelism,
            #     error=error,
            # )
            # scale_factor = 1  # 设置适当的scale_factor
            sram_error_simulator = True
            # apply_sram_error_to_attention(child, sram_error_simulator, parallelism,
            #             error,)

        else:
            convet_sram_op(
                child,
                layer_counter,
                device,
                a_bits=a_bits,
                w_bits=w_bits,
                backend=backend, 
                parallelism=parallelism,
                error=error,
            )


def convert_to_sram_prepare(
    model, 
    inplace=False, 
    device='cpu', 
    a_bits=8, 
    w_bits=8, 
    backend='SRAM', 
    parallelism=128,
    error=4,
    ):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    convet_sram_op(
        model,
        layer_counter,
        device=device,
        a_bits=a_bits,
        w_bits=w_bits,
        backend=backend, 
        parallelism=parallelism,
        error=error,
    )
    return model