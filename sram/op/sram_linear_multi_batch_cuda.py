import time
import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply_cuda import SRAMMultiply

class SRAMLinear(SRAMMultiply):
    """
    SRAMLinear class for performing SRAM-based linear transformation.

    Parameters:
    - N: Total number of bits in input_vector
    - n: Number of bits per input split
    - J: Total number of bits in weights
    - j: Number of bits per weight split
    - m: Number of output neurons
    - k: Bit precision for intermediate values
    - K: Bit precision for final output
    """

    def __init__(self, N, n, J, j, m, k, K):
        super().__init__(N, n, J, j, m, k, K)

    def linear(self, input_vector, weights, noise_sram):
        """
        Perform SRAM-based linear transformation.

        Parameters:
        - input_vector: Input vector as a 2D array (batch_size, input_size)
        - weights: Weight matrix as a 2D array (output_size, input_size)
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a 2D array (batch_size, output_size)
        """
        batch_size, input_size = input_vector.shape
        output_size, _ = weights.shape

        output = torch.zeros((batch_size, output_size)).cuda()

        for b in range(batch_size):
            for i in range(output_size):
                input_values = input_vector[b, :]
                conv_output = self.multiply(input_values, weights[i], noise_sram)
                output[b, i] = torch.sum(conv_output)

        return output

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 20
    noise_sram = torch.randn(N // n, J // j) * 0.01
    input_vector = torch.randint(0, 2 ** N, size=(64, 256)).cuda()
    weights = torch.randint(0, 2 ** J, size=(10, 256)).cuda()

    sram_linear = SRAMLinear(N, n, J, j, weights.shape[-1], k, K)
    start = time.time()
    output = sram_linear.linear(input_vector, weights, noise_sram)
    output_no_noise = sram_linear.linear(input_vector, weights, noise_sram=None)
    end = time.time()
    print("Average time per run:", (end - start) / 2)
    output_tensor = F.linear(input_vector.to(torch.float), weights.to(torch.float))

    print('Input Shape:', input_vector.shape)
    print('Weights Shape:', weights.shape)
    print('Output Shape:', output.shape)
    print('Torch Result:', output_tensor)
    print('Our Result (without noise):', output_no_noise)
    print('Our Result (with noise):', output)

    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor
    print('Our Linear == Torch Linear?', tensor_all_true.all().item())
