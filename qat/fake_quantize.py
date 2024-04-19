import torch
import torch.nn as nn
from .qat import Conv2d_Q
from .qat import Linear_Q
import copy

def convet_qat_op(module, layer_counter, device, a_bits=8, w_bits=8,):
    for name, child in module.named_children():      
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    qat_conv = Conv2d_Q(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=True,
                        a_bits=a_bits,
                        w_bits=w_bits,
                    )
                    qat_conv.bias.data = torch.nn.Parameter(child.bias)
                else:
                    qat_conv = Conv2d_Q(
                        child.in_channels,
                        child.out_channels,
                        child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=False,
                        a_bits=a_bits,
                        w_bits=w_bits,
                    )
            
                qat_conv.weight.data = torch.nn.Parameter(child.weight)
                qat_conv.to(device)
                module._modules[name] = qat_conv

        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] > 1:
                if child.bias is not None:
                    qat_linear = Linear_Q(
                        child.in_features,
                        child.out_features,
                        bias=True,
                        a_bits=a_bits,
                        w_bits=w_bits,
                    )
                    qat_linear.bias.data = torch.nn.Parameter(child.bias)
                else:
                    qat_linear = Linear_Q(
                        child.in_features,
                        child.out_features,
                        bias=False,
                        a_bits=a_bits,
                        w_bits=w_bits,
                    )
                qat_linear.weight.data = torch.nn.Parameter(child.weight)
                qat_linear.to(device)
                module._modules[name] = qat_linear
        else:
            convet_qat_op(
                child,
                layer_counter,
                device,
                a_bits=a_bits,
                w_bits=w_bits,
            )


def fake_quantize_prepare(
    model, 
    inplace=False, 
    device='cpu', 
    a_bits=8, 
    w_bits=8, 
    ):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    convet_qat_op(
        model,
        layer_counter,
        device=device,
        a_bits=a_bits,
        w_bits=w_bits,
    )
    return model