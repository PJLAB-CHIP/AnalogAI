import time
import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply import SRAMMultiply

class SRAMConv2D(SRAMMultiply):
    """
    SRAMConv2D class for performing SRAM-based 2D convolution.

    Parameters:
    - N: Total number of bits in input_values
    - n: Number of bits per input split
    - J: Total number of bits in weights
    - j: Number of bits per weight split
    - m: Number of output neurons
    - k: Bit precision for intermediate values
    - K: Bit precision for final output
    """

    def __init__(self, N, n, J, j, m, k, K):
        super().__init__(N, n, J, j, m, k, K)

    def conv2d(self, input_signal, weights, noise_sram=None):
        """
        Perform SRAM-based 2D convolution.

        Parameters:
        - input_signal: Input signal as a 4D array (batch_size, channels, height, width)
        - weights: Weight matrix as a 4D array (output_channels, input_channels, kernel_height, kernel_width)
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a 4D array (batch_size, output_channels, new_height, new_width)
        """
        batch_size, input_channels, H, W = input_signal.shape
        output_channels, _, K, _ = weights.shape
        m = K ** 2

        new_height = H - K + 1
        new_width = W - K + 1

        output = np.zeros((batch_size, output_channels, new_height, new_width))

        for b in range(batch_size):
            for o_c in range(output_channels):
                for i_c in range(input_channels):
                    input_slice = input_signal[b, i_c, :, :]
                    weight_slice = weights[o_c, i_c, :, :]
                    output_k = np.zeros((new_height, new_width))
                    for i_h in range(new_height):
                        for j_w in range(new_width):
                            # Extract the values within the convolution window and flatten them
                            input_values = input_slice[i_h:i_h + K, j_w:j_w + K].flatten()
                            # Perform SRAM-based multiplication
                            conv_output = self.multiply(input_values, weight_slice.flatten(), noise_sram)
                            output_k[i_h, j_w] = np.sum(conv_output)
                    output[b, o_c, :, :] += output_k

        return output

if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 5
    noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    input_signal = np.random.randint(0, 2 ** N, size=(16, 3, 5, 5))
    weights = np.random.randint(0, 2 ** J, size=(16, 3, 3, 3))

    sram_conv = SRAMConv2D(N, n, J, j, weights.shape[-1] ** 2, k, K)
    start = time.time()
    output = sram_conv.conv2d(input_signal, weights, noise_sram)
    output_no_noise = sram_conv.conv2d(input_signal, weights, noise_sram=None)
    end = time.time()
    print("SRAM average time per run:", (end - start) / 2)
    start1 = time.time()
    output_tensor = F.conv2d(torch.tensor(input_signal), torch.tensor(weights), padding=0)
    end1 = time.time()
    print("torch average time per run:", (end1 - start1))
    print('Input Shape:', input_signal.shape)
    print('Kernel Shape:', weights.shape)
    print('Output Shape:', output.shape)
    print('Output_with_noise:', output)
    print('Output:', output_no_noise)
    print('torch-Output:', output_tensor)

    # Testing against PyTorch Conv2D
    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor
    print('Our Conv2D == Torch Conv2D?', tensor_all_true.all().item())
