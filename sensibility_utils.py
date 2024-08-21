import torch
import torch.nn as nn
import torchvision.models as models
import random
from model.model_set import resnet, vgg, lenet, mobileNetv2, preact_resnet, vit
import numpy as np
import matplotlib.pyplot as plt


def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)


def cosine_schedule(T, beta_min=0.05, beta_max=0.2, s=0.008):
    """
    Cosine noise schedule with specified min and max values.
    
    Args:
    - T: Total number of timesteps.
    - beta_min: Minimum value of beta (noise level).
    - beta_max: Maximum value of beta.
    - s: Small constant added to avoid division by zero.
    
    Returns:
    - betas: An array of betas for each timestep scaled to [beta_min, beta_max].
    """
    def f(t, T, s):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T, s)
    
    for t in range(T + 1):
        alphas.append(f(t, T, s) / f0)
    
    betas = []
    
    for t in range(1, T + 1):
        beta = min(1 - alphas[t] / alphas[t - 1], 0.999)
        betas.append(beta)
    
    betas = np.array(betas)
    
    # Scale betas to [beta_min, beta_max]
    betas = beta_min + (betas - betas.min()) * (beta_max - beta_min) / (betas.max() - betas.min())
    
    return betas

def counter_layer(model):
    layer_counter = 0
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            layer_counter += 1
    return layer_counter


def assign_coefficients_to_layers(model, min_value=0.05, max_value=0.2):
    coefficients = {}

    layer_counter = counter_layer(model)
    coefficients_schedule = cosine_schedule(layer_counter, min_value, max_value) #余弦
    coefficients_schedule = generate_linear_schedule(layer_counter, min_value, max_value) #线性

    layer_idx = 0
    #layer_counter=counter_layer(model)
    # 遍历模型的每一层
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            # 生成范围内的随机系数
            #coefficient = random.uniform(min_value, max_value)
            #线性
            coefficient = coefficients_schedule[layer_idx]
            coefficients[name] = coefficient
            layer_idx += 1

            #余弦
            #coefficient= generate_cosine_values(layer_counter, 0.05, 0.2)

            #print(name)
    return layer_counter, coefficients


# def generate_cosine_schedule(T, s=0.008): #T 整个调度的总步数，要生成的数据点；  s 小的偏移量，调整余弦函数的形状
#     def f(t, T):
#         return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
#     alphas = []
#     f0 = f(0, T)

#     for t in range(T + 1):
#         alphas.append(f(t, T) / f0)
#     betas = []

#     for t in range(1, T + 1):
#         betas.append(min(1 - alphas[t] / alphas[t - 1], 0.2))
#     return  np.array(betas)


# def generate_cosine_values(num_values, min_value, max_value):
#     # 生成从 0 到 π 的等间距角度
#     angles = np.linspace(0, np.pi, num_values)
    
#     # 计算余弦值，范围在 [1, -1]
#     cosine_values = np.cos(angles)
    
#     # 线性变换，将 [1, -1] 范围映射到 [min_value, max_value]
#     scaled_values = min_value + (cosine_values + 1) * (max_value - min_value) / 2

#     return scaled_values

# import numpy as np






# # Example usage:
# T = 18
# betas = cosine_schedule(T, beta_min=0.05, beta_max=0.2)
# print("Scaled Cosine Betas:", betas)

# Plot the betas
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, T+1), betas, marker='o')
# plt.title('Cosine Schedule with Min-Max Scaling')
# plt.xlabel('Timestep')
# plt.ylabel('Beta Value')
# plt.grid(True)
# plt.show()
# plt.savefig('plot_cos.png')
# # 使用这个函数生成 18 个在 [0.05, 0.2] 范围内的余弦值
# cosine_values = generate_cosine_values(18, 0.05, 0.2)
#print(cosine_values)


# # Example usage:
# T = 18
# co = generate_linear_schedule(T, low=0.05, high=0.2)
# print("Scaled linear co:", co)

# # Plot the betas
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, T+1), co, marker='o')
# plt.title('linear Schedule with Min-Max Scaling')
# plt.xlabel('Timestep')
# plt.ylabel('co Value')
# plt.grid(True)
# plt.show()
# plt.savefig('plot_linear.png')
# # 使用这个函数生成 18 个在 [0.05, 0.2] 范围内的余弦值
# cosine_values = generate_cosine_values(18, 0.05, 0.2)
# #print(cosine_values)


# 使用 ResNet-18 模型
model = resnet.ResNet18(in_channels=1)


layer_counter , coefficients = assign_coefficients_to_layers(model)


# 打印每一层及其对应的系数
for layer_name, coefficient in coefficients.items():
   print(f"Layer: {layer_name}, Coefficient: {coefficient}")


plt.plot(range(1, layer_counter+1), list(coefficients.values()), marker='o')
plt.savefig('2.png')
