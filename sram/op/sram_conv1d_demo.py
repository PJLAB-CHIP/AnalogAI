import torch
import numpy as np
import time
import torch
import torch.nn.functional as F

def clamp(value, bits):
    max_value = np.maximum(2 ** (bits - 1) - 1, value.min())
    min_value = np.minimum(-2 ** (bits - 1), value.max())
    return np.clip(value, min_value, max_value).astype(int)

def sram(input_values, weights, N, n, J, j, m, noise_sram, k, K):
    input_splits = [(input_values >> i) & 0b11 for i in range(0, N, n)]
    weight_splits = [(weights >> i) & 0b1 for i in range(0, J, j)]

    partial_sums = np.zeros((len(input_splits), len(weight_splits), m), dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            partial_product = input_split * weight_split
            partial_product = clamp(partial_product, k)
            partial_sums[i_idx, w_idx, :] += partial_product.astype(int)

    output = np.zeros(m, dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            partial_sum = partial_sums[i_idx, w_idx, :]
            partial_sum = partial_sum << (i_idx * n + w_idx * j)
            partial_sum = clamp(partial_sum, K)
            output += partial_sum

    return np.sum(output)


def one_dim_conv(input_signal, weights, L, l, bits, N, J, n, j):
    k = bits
    K = bits
    noise_sram = 0
    output = []
    for i in range(L - l + 1):
        input_values = input_signal[i:i + l]
        conv_output = sram(input_values, weights, N, n, J, j, l, noise_sram, k, K)
        output.append(conv_output)
    return output

if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    bits = 20
    L = 8
    l = 3
    input_signal = np.random.randint(0, 2 ** N, size=L)
    weights = np.random.randint(0, 2 ** J, size=l)
    output = one_dim_conv(input_signal, weights, L, l, bits, N, J, n, j)
    output_tensor = F.conv1d(torch.tensor(input_signal).view(1, 1, -1), torch.tensor(weights).view(1, 1, -1))
    print('input:', input_signal)
    print('kernel:', weights)
    print('torch reslut:',output_tensor)
    print('our reslut:', output)
    tensor_all_true = torch.tensor(output,dtype=torch.int64) == output_tensor.squeeze(0)
    print('our conv1d==torch_conv1d?',tensor_all_true.all().item())
