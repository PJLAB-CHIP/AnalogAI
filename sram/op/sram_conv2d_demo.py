import torch
import numpy as np
from scipy.signal import convolve2d
import time
import torch
import torch.nn.functional as F
def clamp(value, bits):
    max_value = np.maximum(2 ** (bits - 1) - 1, value.min())
    min_value = np.minimum(-2 ** (bits - 1), value.max())

    return np.clip(value, min_value, max_value).astype(int)

def sramm(input_values, weights, N, n, J, j, m, noise_sram, k, K):
    input_splits = [(input_values >> i) & 0b11 for i in range(0, N, n)]
    weight_splits = [(weights >> i) & 0b1 for i in range(0, J, j)]

    partial_sums = np.zeros((len(input_splits), len(weight_splits), m), dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            if noise_sram is not None:
                partial_product = input_split.flatten()[:k] * weight_split.flatten()[:k] * (1 + noise_sram[i_idx, w_idx])
            else:
                partial_product = input_split.flatten()[:k] * weight_split.flatten()[:k]
            partial_product = clamp(partial_product, k)
            partial_sums[i_idx, w_idx, :] += partial_product

    output = np.zeros(m, dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            partial_sum = partial_sums[i_idx, w_idx, :]
            partial_sum = partial_sum << (i_idx * n + w_idx * j)
            partial_sum = clamp(partial_sum, K)
            output += partial_sum

    output = np.sum(output)
    return output

def two_dim_conv(input_signal, weights, K_bits, k_bits, N, J, n, j, noise_sram):
    H, W = input_signal.shape
    K, K = weights.shape
    m = K**2
    output = np.zeros((H - K + 1, W - K + 1))
    for i_h in range(H - K + 1):
        for j_w in range(W - K + 1):
            input_values = input_signal[i_h:i_h + K, j_w:j_w + K].flatten()
            conv_output = sramm(input_values, weights.flatten(), N, n, J, j, m, noise_sram, k_bits, K_bits)
            output[i_h, j_w] = conv_output

    return output

if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K_bits=20
    k_bits=20
    noise_sram = np.random.normal(0, 0.01, size=(N//n, J//j))
    input_signal = np.random.randint(0, 2 ** N, size=(5,5))
    weights = np.random.randint(0, 2 ** J, size=(3,3))
    output = two_dim_conv(input_signal, weights, K_bits, k_bits, N, J, n, j, noise_sram)
    output_no_noise = two_dim_conv(input_signal, weights, K_bits, k_bits, N, J, n, j, noise_sram=None)   
    output_tensor = F.conv2d(torch.tensor(input_signal).unsqueeze(0).unsqueeze(0), torch.tensor(weights).unsqueeze(0).unsqueeze(0), padding=0)
    print('input:', input_signal)
    print('kernel:', weights)
    print('torch reslut:',output_tensor)
    print('our reslut:', output_no_noise)
    print('when add noise, our result:', output)
    tensor_all_true = torch.tensor(output_no_noise,dtype=torch.int64) == output_tensor.squeeze(0).squeeze()
    print('our two_dim_conv == torch_conv2d?',tensor_all_true.all().item())
