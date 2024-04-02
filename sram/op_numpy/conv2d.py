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

    def conv2d(self, input_signal, 
               weights, 
               stride, 
               padding, 
               groups, 
               bias=None, 
               noise_sram=None):
        """
        Perform SRAM-based 2D convolution.

        Parameters:
        - input_signal: Input signal as a 4D array (batch_size, channels, height, width)
        - weights: Weight matrix as a 4D array (output_channels, input_channels, kernel_height, kernel_width)
        - stride: Stride for convolution
        - padding: The count of zeros to pad on both sides
        - dilation: The space between kernel elements
        - groups: Split the input to groups
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a 4D array (batch_size, output_channels, new_height, new_width)
        """
        batch_size, input_channels, H, W = input_signal.shape
        output_channels, k_channels, K, K1 = weights.shape

        assert (K == K1)
        assert (input_channels % groups == 0)
        assert (output_channels % groups == 0)
        assert (input_channels // groups == k_channels)
        if bias is not None:
            assert (bias.shape[0] == output_channels)


        new_height = (H + 2 * padding - K) // stride + 1
        new_width = (W + 2 * padding - K) // stride + 1

        input_pad = np.pad(input_signal, [(0, 0), (0, 0), (padding, padding), (padding, padding)])

        output = np.zeros((batch_size, output_channels, new_height, new_width))

        c_o_per_group = output_channels // groups

        for b in range(batch_size):
            for i_h in range(new_height):
                for i_w in range(new_width):
                    for i_c in range(output_channels):
                        i_g = i_c // c_o_per_group
                        h_lower = i_h * stride
                        h_upper = i_h * stride + K
                        w_lower = i_w * stride
                        w_upper = i_w * stride + K
                        c_lower = i_g * k_channels
                        c_upper = (i_g + 1) * k_channels
                        input_slice = input_pad[b, c_lower:c_upper, h_lower:h_upper, w_lower:w_upper].flatten()
                        weight_slice = weights[i_c].flatten()
                        conv_output = self.calculate(input_slice, weight_slice, noise_sram)
                        output[b, i_c, i_h, i_w] = np.sum(conv_output)
                        if bias:
                            output[b, i_c, i_h, i_w] += bias[i_c]
        return output

if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 5
    m = 32
    groups = 3 
    stride = 2  
    padding = 1  

    # noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    noise = np.round(np.random.normal(0, 1))
    input_signal = np.random.randint(0, 2 ** N, size=(32, 12, 5, 5))
    weights = np.random.randint(0, 2 ** J, size=(6, 4, 3, 3))

    sram_conv = SRAMConv2D(N, n, J, j, m, k, K)
    start = time.time()
    output = sram_conv.conv2d(input_signal, weights, stride, padding, groups, noise_sram=noise)
    output_no_noise = sram_conv.conv2d(input_signal, weights, stride, padding, groups, noise_sram=None)
    end = time.time()
    print("SRAM average time per run:", (end - start)/2)
    output_tensor = F.conv2d(torch.tensor(input_signal), torch.tensor(weights), groups=groups, stride=2, padding=1)
    print('Input Shape:', input_signal.shape)
    print('Kernel Shape:', weights.shape)
    print('Output Shape:', output.shape)
    print('Output_with_noise:', output)
    print('Output:', output_no_noise)
    print('torch-Output:', output_tensor)
    # Testing against PyTorch Conv2D
    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor
    print('Our Conv2D == Torch Conv2D?', tensor_all_true.all().item())