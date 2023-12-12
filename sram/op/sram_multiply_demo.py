import numpy as np
import time
import torch

def clamp(value, bits):
    max_value = np.maximum(2 ** (bits - 1) - 1, value.min())
    min_value = np.minimum(-2 ** (bits - 1), value.max())

    return np.clip(value, min_value, max_value).astype(int)

def sram(input_values, weights, N, n, J, j, m, k, K, noise_sram=None):
    input_splits = [(input_values >> i) & 0b11 for i in range(0, N, n)]
    weight_splits = [(weights >> i) & 0b1 for i in range(0, J, j)]

    partial_sums = np.zeros((len(input_splits), len(weight_splits), m), dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            if noise_sram is None:
                partial_product = input_split * weight_split
            else:
                partial_product = input_split * weight_split * (1 + noise_sram[i_idx, w_idx])
            partial_product = clamp(partial_product, k)
            partial_sums[i_idx, w_idx, :] += partial_product.astype(int)

    output = np.zeros(m, dtype=int)
    for i_idx, input_split in enumerate(input_splits):
        for w_idx, weight_split in enumerate(weight_splits):
            partial_sum = partial_sums[i_idx, w_idx, :]
            partial_sum = partial_sum << (i_idx * n + w_idx * j)
            partial_sum = clamp(partial_sum, K)
            output += partial_sum
    #output = np.sum(output)
    return output



if __name__ == "__main__":
    N = 8
    n = 2
    J = 8
    j = 1
    K = 20
    k = 5
    m = 32
    noise_sram = np.random.normal(0, 0.01, size=(N//n, J//j))
    start = time.time()
    run = 1000
    input_values = np.random.randint(0, 2 ** N, size=m)
    weights = np.random.randint(0, 2 ** J, size=m)   
    output = sram(input_values, weights, N, n, J, j, m, k, K, noise_sram=None)
    end = time.time()
    print((end-start)/run)
    target_output = input_values * weights
    print("Input values:", input_values)
    print("Weights:", weights)
    print("Output:", output)
    print("Target_output:",target_output)
    print(target_output==output)