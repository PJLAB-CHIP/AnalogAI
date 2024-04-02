import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply import SRAMMultiply

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
        - input_vector: Input vector as a 1D array
        - weights: Weight matrix as a 2D array
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a 1D array
        """
        L = len(input_vector)
        l, _ = weights.shape  # Assuming weights is a 2D array

        output = np.zeros(l,)

        for i in range(l):
            input_values = input_vector
            conv_output = self.multiply(input_values, weights[i], noise_sram)
            output[i] = np.sum(conv_output)

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
    noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    input_vector = np.random.randint(0, 2 ** N, size=(8,))
    weights = np.random.randint(0, 2 ** J, size=(3,8))

    sram_linear = SRAMLinear(N, n, J, j, weights.shape[-1], k, K)

    output = sram_linear.linear(input_vector, weights, noise_sram)
    output_no_noise = sram_linear.linear(input_vector, weights, noise_sram=None)
    output_tensor = F.linear(torch.tensor(input_vector).view(1, -1), torch.tensor(weights))

    print('Input:', input_vector)
    print('Weights:', weights)
    print('Torch Result:', output_tensor.squeeze().numpy())
    print('Our Result (without noise):', output_no_noise)
    print('Our Result (with noise):', output)

    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor.squeeze()
    print('Our Linear == Torch Linear?', tensor_all_true.all().item())
