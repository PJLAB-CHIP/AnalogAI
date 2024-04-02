import numpy as np
import time

class SRAMMultiply:
    def __init__(self, N, n, J, j, m, k, K):
        """
        SRAMMultiply class for performing SRAM-based multiplication.

        Parameters:
        - N: Total number of bits in input_values
        - n: Number of bits per input split
        - J: Total number of bits in weights
        - j: Number of bits per weight split
        - m: Parallelism of computation
        - k: Bit precision for intermediate values
        - K: Bit precision for final output
        """
        self.N = N
        self.n = n
        self.J = J
        self.j = j
        self.m = m
        self.k = k
        self.K = K

    def clamp(self, value, bits):
        """
        Clamp the input value within the specified number of bits.

        Parameters:
        - value: Input value to be clamped
        - bits: Number of bits for clamping

        Returns:
        - Clamped value as an integer
        """
        max_value = np.maximum(2 ** (bits - 1) - 1, value.min())
        min_value = np.minimum(-2 ** (bits - 1), value.max())
        return np.clip(value, min_value, max_value).astype(int)

    def multiply(self, input_values, weights, noise_sram=None):
        """
        Perform SRAM-based multiplication.

        Parameters:
        - input_values: Input values as an array
        - weights: Weight values as an array
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as an array
        """
        input_splits = [(input_values >> i) & 0b11 for i in range(0, self.N, self.n)]
        weight_splits = [(weights >> i) & 0b1 for i in range(0, self.J, self.j)]

        partial_sums = np.zeros((len(input_splits), len(weight_splits), self.m), dtype=int)

        for i_idx, input_split in enumerate(input_splits):
            for w_idx, weight_split in enumerate(weight_splits):
                partial_product = input_split * weight_split
                partial_product = self.clamp(partial_product, self.k)
                partial_sums[i_idx, w_idx, :] += partial_product.astype(int)

        output = np.zeros(self.m, dtype=int)
        for i_idx, input_split in enumerate(input_splits):
            for w_idx, weight_split in enumerate(weight_splits):
                partial_sum = partial_sums[i_idx, w_idx, :]
                partial_sum = partial_sum << (i_idx * self.n + w_idx * self.j)
                partial_sum = self.clamp(partial_sum, self.K)
                output += partial_sum

        if noise_sram is not None:
            return np.sum(output) + int(noise_sram)
        else:
            return np.sum(output)
    
    def calculate(self, input_values, weights, noise_sram=None):
        num_chunks = int(np.ceil(len(input_values) / self.m))
        padded_input = np.pad(input_values, (0, num_chunks * self.m - len(input_values)))
        input = padded_input.reshape((num_chunks, self.m))
        padded_weight = np.pad(weights, (0, num_chunks * self.m - len(weights)))
        weight = padded_weight.reshape((num_chunks, self.m))
        cal = 0
        for i in range(num_chunks):
            cal += self.multiply(input[i], weight[i], noise_sram)

        # Reshape the padded input into chunks of size chunk_size
        return cal


if __name__ == "__main__":
    N = 8
    n = 2
    J = 8
    j = 1
    K = 20
    k = 5
    m = 32
    noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    noise = np.round(np.random.normal(0, 1))
    print(noise)
    run = 1000

    sram_processor = SRAMMultiply(N, n, J, j, m, k, K)

    start = time.time()
    for _ in range(run):
        input_values = np.random.randint(0, 2 ** N, size=m+10)
        weights = np.random.randint(0, 2 ** J, size=m+10)
        output = sram_processor.calculate(input_values, weights, noise_sram=None)
    end = time.time()

    print("Average time per run:", (end - start) / run)
    target_output = np.sum(input_values * weights)
    noise_output = sram_processor.calculate(input_values, weights, noise_sram=noise)
    print("Input values:", input_values)
    print("Weights:", weights)
    print('noise output:', noise_output)
    print("Output:", output)
    print("Target_output:", target_output)
    print("Outputs are equal:", np.array_equal(target_output, output))
