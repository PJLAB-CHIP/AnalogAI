import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

# model = timm.create_model('resnet18',pretrained=True,num_classes=10)
# # print(model)
# model_names = timm.list_models(pretrained=True)
# print(model_names)
# w = np.linspace(0., 0.02, num=5, endpoint=True)
# print(w)
# import timm

# # 获取所有 Vision Transformer 模型的名称
# all_models = timm.list_models('*vit*')

# # 打印模型列表
# print(all_models)
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
import torch
from model import resnet_Q, resnet
# 定义一个简单的 PyTorch 模型
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

# 创建模型实例
model = resnet.ResNet50()

# 对模型进行符号化跟踪
traced_model = symbolic_trace(model)

# 打印符号化跟踪后的模型
# print(traced_model.graph)
print(traced_model.code)
# for node in traced_model.graph.nodes:
#     print(node.op)
# traced_model.graph.print_tabular()
