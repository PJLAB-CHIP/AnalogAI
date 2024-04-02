import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply import SRAMMultiply


import numpy as np

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

    def conv2d(self, input_signal, weights, noise_sram):
        """
        Perform SRAM-based 2D convolution.

        Parameters:
        - input_signal: Input signal as a 2D array
        - weights: Weight matrix as a 2D array
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a 2D array
        """
        H, W = input_signal.shape
        K, K = weights.shape
        m = K**2

        # Initialize the output array with zeros
        output = np.zeros((H - K + 1, W - K + 1))

        # Iterate over the input signal with the convolution window
        for i_h in range(H - K + 1):
            for j_w in range(W - K + 1):
                # Extract the values within the convolution window and flatten them
                input_values = input_signal[i_h:i_h + K, j_w:j_w + K].flatten()

                # Perform SRAM-based multiplication
                conv_output = self.multiply(input_values, weights.flatten(), noise_sram)

                # Sum the products and store in the output array
                output[i_h, j_w] = np.sum(conv_output)

        return output


if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 4
    noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    input_signal = np.random.randint(0, 2 ** N, size=(5, 5))
    weights = np.random.randint(0, 2 ** J, size=(3, 3))

    sram_conv = SRAMConv2D(N, n, J, j, weights.shape[0]**2, k, K)

    output = sram_conv.conv2d(input_signal, weights, noise_sram)
    output_no_noise = sram_conv.conv2d(input_signal, weights, noise_sram=None)
    output_tensor = F.conv2d(torch.tensor(input_signal).unsqueeze(0).unsqueeze(0), torch.tensor(weights).unsqueeze(0).unsqueeze(0), padding=0)

    print('Input:', input_signal)
    print('Kernel:', weights)
    print('Torch Result:', output_tensor)
    print('Our Result (without noise):', output_no_noise)
    print('Our Result (with noise):', output)

    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor.squeeze(0).squeeze()
    print('Our Conv2D == Torch Conv2D?', tensor_all_true.all().item())