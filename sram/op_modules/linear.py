import time
import numpy as np
import torch
import torch.nn.functional as F


class SRAMLinear():
    """
    SRAMLinear class for performing SRAM-based linear transformation.
    
    Parameters:
    - parallelism: Degree of parallelism in MAC operations.
    - error_range: The range of random error to be introduced.
    """

    def __init__(self, m, error):
        # super().__init__(m, error)
        self.parallelism = m
        self.error_range = error

    def matrix_multiply_with_error(self, input, weight, parallelism=32, error_range=2):
        """
        Perform matrix multiplication with random error introduced in MAC operations.

        Parameters:
        - input: Input matrix.
        - weight: Weight matrix (transposed).
        - parallelism: Degree of parallelism in MAC operations.
        - error_range: The range of random error to be introduced.

        Returns:
        - The result of matrix multiplication with random errors.
        """
        output_rows, input_cols = input.shape
        weight_rows = weight.shape[0]

        # Perform matrix multiplication
        output = np.dot(input, weight.T)

        # Compute the number of MAC operations per output element
        mac_per_output_element = input_cols

        # Compute the number of errors introduced per output element
        errors_per_output_element = mac_per_output_element // parallelism

        # Generate and add random errors
        total_errors = np.zeros(output.shape).astype(np.int64)
        for _ in range(errors_per_output_element):
            random_error = np.round(np.random.uniform(-error_range, error_range, output.shape)).astype(np.int64)
            total_errors += random_error
    
        output += total_errors

        return output

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
        if noise_sram is not None:
            linear_out = self.matrix_multiply_with_error(input_vector, weights, parallelism=self.parallelism, error_range=self.error_range)
        else:
            linear_out = np.dot(input_vector, weights.T)

        return linear_out

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 4
    m = 32
    error = 10
    # noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    noise_sram = True
    input_vector = np.random.randint(0, 2 ** (N-1), size=(256, 1024))
    weights = np.random.randint(0, 2 ** (J-1), size=(10, 1024))

    sram_linear = SRAMLinear(m=m,error=error)

    start0 = time.time()
    output_tensor = F.linear(torch.tensor(input_vector, dtype=torch.int64), torch.tensor(weights, dtype=torch.int64))
    start = time.time()
    output = sram_linear.linear(input_vector, weights, noise_sram)
    output_no_noise = sram_linear.linear(input_vector, weights, noise_sram=None)
    end = time.time()
    print("SRAM average time per run:", (end - start) / 2)
    print("torch average time per run:", (start - start0))
    print('Input Shape:', input_vector.shape)
    print('Weights Shape:', weights.shape)
    print('Output Shape:', output.shape)
    print('Torch Result:', output_tensor.numpy())
    print('Our Result (without noise):', output_no_noise)
    print('Our Result (with noise):', output)

    tensor_all_true = torch.tensor(output_no_noise, dtype=torch.int64) == output_tensor
    print('Our Linear == Torch Linear?', tensor_all_true.all().item())
