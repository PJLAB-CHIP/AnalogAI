import torch
import torch.nn.functional as F
from sram_multiply_cuda import SRAMMultiply

import time
import numpy as np

class SRAMConv2DCUDA(SRAMMultiply):
    def __init__(self, N, n, J, j, m, k, K):
        super().__init__(N, n, J, j, m, k, K)

    def conv2d(self, input_signal, weights, noise_sram=None):
        batch_size, input_channels, H, W = input_signal.shape
        output_channels, _, K, _ = weights.shape
        m = K ** 2

        new_height = H - K + 1
        new_width = W - K + 1

        # Initialize the output array with zeros
        output = torch.zeros((batch_size, output_channels, new_height, new_width), device='cuda')

        # Move input_signal, weights, and noise_sram to GPU if available
        input_signal_cuda = input_signal
        weights_cuda = weights
        if noise_sram is not None:
            noise_sram_cuda = noise_sram
        else:
            noise_sram_cuda = None

        for b in range(batch_size):
            for o_c in range(output_channels):
                for i_c in range(input_channels):
                    input_slice = input_signal_cuda[b, i_c, :, :]
                    weight_slice = weights_cuda[o_c, i_c, :, :]
                    output_k = torch.zeros((new_height, new_width), device='cuda')
                    for i_h in range(new_height):
                        for j_w in range(new_width):
                            input_values = input_slice[i_h:i_h + K, j_w:j_w + K].flatten()
                            conv_output = self.multiply(input_values, weight_slice.flatten(), noise_sram_cuda)
                            output_k[i_h, j_w] = torch.sum(conv_output)
                    output[b, o_c, :, :] += output_k

        return output.cpu().numpy()  # Move the result back to CPU

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 5
    noise_sram = torch.randn(N // n, J // j) * 0.01
    input_vector = torch.randint(0, 2 ** N, size=(1, 4, 5, 5)).cuda()
    weights = torch.randint(0, 2 ** J, size=(2, 4, 3, 3)).cuda()

    sram_conv = SRAMConv2DCUDA(N, n, J, j, weights.shape[-1] ** 2, k, K)
    start = time.time()
    output = sram_conv.conv2d(input_vector, weights, noise_sram)
    output_no_noise = sram_conv.conv2d(input_vector, weights, noise_sram=None)
    end = time.time()
    print("Average time per run:", (end - start) / 2)
    output_tensor = F.conv2d(input_vector.to(torch.float), weights.to(torch.float), padding=0)

    print('Input Shape:', input_vector.shape)
    print('Kernel Shape:', weights.shape)
    print('Output Shape:', output.shape)
    print('Output_with_noise:', output)
    print('Output:', output_no_noise)
    print('torch-Output:', output_tensor)

    # Testing against PyTorch Conv2D
    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64).cuda() == output_tensor
    print('Our Conv2D == Torch Conv2D?', tensor_all_true.all().item())
