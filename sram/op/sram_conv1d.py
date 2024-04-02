import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply import SRAMMultiply

class SRAMConv1D(SRAMMultiply):
    def __init__(self, N, n, J, j, m, k, K):
        super().__init__(N, n, J, j, m, k, K)

    def conv1d(self, input_signal, weights, noise_sram):
        """
        Perform one-dimensional convolution using SRAM-based multiplication.

        Parameters:
        - input_signal: Input signal as an array
        - weights: Kernel weights as an array
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed convolution output as a list
        """
        output = []
        L = input_signal.shape[0]
        l = weights.shape[0]
        for i in range(L - l + 1):
            input_values = input_signal[i:i + l]
            conv_output = self.multiply(input_values, weights, noise_sram)
            output.append(np.sum(conv_output))
        
        return output

if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 20
    L = 8
    l = 3

    # noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    noise_sram = None
    input_signal = np.random.randint(0, 2 ** N, size=L)
    weights = np.random.randint(0, 2 ** J, size=l)

    sram_conv = SRAMConv1D(N, n, J, j, weights.shape[0], k, K)

    output = sram_conv.conv1d(input_signal, weights, noise_sram)
    output_tensor = F.conv1d(torch.tensor(input_signal).view(1, 1, -1), torch.tensor(weights).view(1, 1, -1))

    print('Input:', input_signal)
    print('Kernel:', weights)
    print('Torch Result:', output_tensor)
    print('Our Result:', output)

    tensor_all_true = torch.tensor(output, dtype=torch.int64) == output_tensor.squeeze(0)
    print('Our Conv1D == Torch Conv1D?', tensor_all_true.all().item())