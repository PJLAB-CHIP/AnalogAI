import torch
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
        - m: Number of output neurons
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
        - Clamped value as a PyTorch tensor
        """
        max_value = torch.maximum(torch.tensor(2 ** (bits - 1) - 1),value.min().clone().detach())
        min_value = torch.minimum(torch.tensor(-2 ** (bits - 1)), value.max().clone().detach())
        return torch.clip(value, min_value, max_value).to(torch.int)

    def multiply(self, input_values, weights, noise_sram=None):
        """
        Perform SRAM-based multiplication.

        Parameters:
        - input_values: Input values as a PyTorch tensor
        - weights: Weight values as a PyTorch tensor
        - noise_sram: SRAM noise matrix (optional)

        Returns:
        - Computed output as a PyTorch tensor
        """
        input_splits = [(input_values >> i) & 0b11 for i in range(0, self.N, self.n)]
        weight_splits = [(weights >> i) & 0b1 for i in range(0, self.J, self.j)]

        partial_sums = torch.zeros((len(input_splits), len(weight_splits), self.m), dtype=torch.int).cuda()

        for i_idx, input_split in enumerate(input_splits):
            for w_idx, weight_split in enumerate(weight_splits):
                if noise_sram is None:
                    partial_product = input_split * weight_split
                else:
                    partial_product = input_split * weight_split * (1 + noise_sram[i_idx, w_idx])
                partial_product = self.clamp(partial_product, self.k)
                partial_sums[i_idx, w_idx, :] += partial_product

        output = torch.zeros(self.m, dtype=torch.int).cuda()
        for i_idx, input_split in enumerate(input_splits):
            for w_idx, weight_split in enumerate(weight_splits):
                partial_sum = partial_sums[i_idx, w_idx, :]
                partial_sum = partial_sum << (i_idx * self.n + w_idx * self.j)
                partial_sum = self.clamp(partial_sum, self.K)
                output += partial_sum

        return output

if __name__ == "__main__":
    N = 8
    n = 2
    J = 8
    j = 1
    K = 20
    k = 5
    m = 32
    noise_sram = torch.randn(N // n, J // j) * 0.01
    run = 1000

    sram_processor = SRAMMultiply(N, n, J, j, m, k, K)

    start = time.time()
    for _ in range(run):
        input_values = torch.randint(0, 2 ** N, size=(m,)).cuda()
        weights = torch.randint(0, 2 ** J, size=(m,)).cuda()
        output = sram_processor.multiply(input_values, weights, noise_sram=None)
    end = time.time()

    print("Average time per run:", (end - start) / run)
    target_output = input_values * weights
    print("Input values:", input_values)
    print("Weights:", weights)
    print("Output:", output)
    print("Target_output:", target_output)
    print("Outputs are equal:", torch.equal(target_output, output))
