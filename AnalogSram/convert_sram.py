import torch
import torch.nn as nn
import torch.nn.functional as F
from .sram_op import SramConv2d
from .sram_op import SramLinear
from .sram_op import apply_sram_error_to_attention, sram_einsum
from model.vit import MultiHeadSelfAttention
from torch import distributed
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.autograd import Function
import math
import copy



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
            child.parallelism = parallelism
            child.error = error
            # apply_sram_error_to_attention(child, sram_error_simulator, parallelism,
            #             error)

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