import time
import numpy as np
import torch
import torch.nn.functional as F
from sram_multiply import SRAMMultiply


import numpy as np

class SRAMConv2D():
    """
    SRAMConv2D class for performing SRAM-based 2D convolution.

    Parameters:
    - parallelism: Degree of parallelism in MAC operations.
    - error_range: The range of random error to be introduced.
    """

    def __init__(self, m, error):
        # super().__init__(m, error)
        self.parallelism = m
        self.error_range = error

    def generate_random_errors(num_errors, error_range, shape):
        """
        Generate random errors for each element in the matrix.
        Each element will have 'num_errors' random errors added to it.
        """
        total_errors = np.zeros(shape)
        for _ in range(num_errors):
            random_error = np.round(np.random.uniform(-error_range, error_range, shape))
            total_errors += random_error
        return total_errors

    def matrix_multiply_with_error(self, input_col, weight_col, parallelism=32, error_range=2):
        """
        Perform matrix multiplication with random error introduced in MAC operations.

        Parameters:
        - input_col: Input matrix.
        - weight_col: Weight matrix (transposed).
        - parallelism: Degree of parallelism in MAC operations.
        - error_range: The range of random error to be introduced.

        Returns:
        - The result of matrix multiplication with random errors.
        """
        output_rows, input_cols = input_col.shape
        weight_rows = weight_col.shape[0]

        # Perform matrix multiplication
        output = np.dot(input_col, weight_col.T)

        # Compute the number of MAC operations per output element
        mac_per_output_element = input_cols

        # Compute the number of errors introduced per output element
        errors_per_output_element = mac_per_output_element // parallelism

        # Generate and add random errors
        total_errors = np.zeros(output.shape)
        for _ in range(errors_per_output_element):
            random_error = np.round(np.random.uniform(-error_range, error_range, output.shape))
            total_errors += random_error
    
        output += total_errors

        return output

    def img2col(self, input_data, kernel_height, kernel_width, stride, padding):
        """
        Apply img2col operation to transform the input data into columns.

        Parameters:
        - input_data: The input data
        - kernel_height: The height of the kernel
        - kernel_width: The width of the kernel
        - stride: The stride of the convolution
        - padding: The padding applied to the input data


        Returns:
        - A 2D array where each row is a flattened convolution window
        """
        N, C, H, W = input_data.shape
        out_height = (H - kernel_height) // stride + 1
        out_width = (W - kernel_width) // stride + 1
        img_col = np.zeros((N, C, kernel_height, kernel_width, out_height, out_width))

        for y in range(kernel_height):
            y_max = y + stride * out_height
            for x in range(kernel_width):
                x_max = x + stride * out_width
                img_col[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]

        # Reshape and transpose to get the desired 2D array
        return img_col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_height * out_width, -1)

    def conv2d(self, input_signal, weights, stride, padding, groups, bias=None, noise_sram=None):
        """
        Perform SRAM-based 2D convolution using img2col optimization.

        Parameters:
        - input_signal: Input signal as a 4D array (batch_size, channels, height, width)
        - weights: Weight matrix as a 4D array (output_channels, input_channels, kernel_height, kernel_width)
        - stride: Stride for convolution
        - padding: The count of zeros to pad on both sides
        - groups: Split the input to groups
        - bias: Optional bias for each output channel
        - noise_sram: Optional SRAM noise matrix

        Returns:
        - Computed output as a 4D array (batch_size, output_channels, new_height, new_width)
        """

        batch_size, input_channels, H, W = input_signal.shape
        output_channels, k_channels, kernel_height, kernel_width = weights.shape
        group_input_channels = input_channels // groups
        group_output_channels = output_channels // groups
        
        assert (kernel_height == kernel_width)
        assert (input_channels % groups == 0)
        assert (output_channels % groups == 0)
        assert (input_channels // groups == k_channels)
        if bias is not None:
            assert (bias.shape[0] == output_channels)

        # Computing new height and width after convolution
        new_height = (H + 2 * padding - kernel_height) // stride + 1
        new_width = (W + 2 * padding - kernel_width) // stride + 1

        input_pad = np.pad(input_signal, [(0, 0), (0, 0), (padding, padding), (padding, padding)])
        output = np.zeros((batch_size, output_channels, new_height, new_width))

        for g in range(groups):

            group_input = input_pad[:, g * group_input_channels:(g + 1) * group_input_channels, :, :]
            group_weights = weights[g * group_output_channels:(g + 1) * group_output_channels, :, :, :]

            # Apply img2col 
            input_col = self.img2col(group_input, kernel_height, kernel_width, stride, padding)
            weight_col = group_weights.reshape(group_output_channels, -1)

            # Perform matrix multiplication
            if noise_sram is not None:
                conv_out = self.matrix_multiply_with_error(input_col, weight_col, parallelism=self.parallelism, error_range=self.error_range)
            else:
                conv_out = np.dot(input_col, weight_col.T)

            conv_out = conv_out.reshape(batch_size, new_height, new_width, group_output_channels)
            output[:, g * group_output_channels:(g + 1) * group_output_channels, :, :] = conv_out.transpose(0, 3, 1, 2)

        # Add bias
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)

        return output


if __name__ == "__main__":
    N = 8
    J = 8
    n = 2
    j = 1
    K = 20
    k = 5
    m = 32
    error = 2
    groups = 1 
    stride = 2  
    padding = 1  


    # noise_sram = np.random.normal(0, 0.01, size=(N // n, J // j))
    # noise = True
    noise = True
    input_signal = np.random.randint(0, 2 ** N, size=(32, 256, 64, 64))
    weights = np.random.randint(0, 2 ** J, size=(512, 256, 3, 3))

    sram_conv = SRAMConv2D(m=m,error=error)
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