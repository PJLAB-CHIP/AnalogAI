import torch
import torch.nn.functional as F

class SRAMConv2d():
    """
    SRAMConv2d class for performing 2D convolution operations optimized for SRAM.
    
    This class implements a custom 2D convolution operation that is optimized for SRAM-based
    systems, especially focusing on INT8 data types for efficient computation.
    """

    def __init__(self):
        """
        Initializes the SRAMConv2d class.
        """
        super().__init__()

    def img2col(self, input_data, kernel_height, kernel_width, stride, padding):
        """
        Transforms the input data into columns for efficient convolution operation.
        
        The img2col operation is a common technique in convolution operations that converts
        the input data into a 2D array where each row is a flattened convolution window.

        Parameters:
        - input_data (torch.Tensor): The input tensor in the shape (N, C, H, W).
        - kernel_height (int): The height of the convolution kernel.
        - kernel_width (int): The width of the convolution kernel.
        - stride (int): The stride of the convolution operation.
        - padding (int): The amount of zero-padding added to both sides of the input.

        Returns:
        - torch.Tensor: A 2D tensor where each row is a flattened convolution window.
        """
        N, C, H, W = input_data.shape
        out_height = (H + 2 * padding - kernel_height) // stride + 1
        out_width = (W + 2 * padding - kernel_width) // stride + 1

        input_padded = F.pad(input_data, (padding, padding, padding, padding))
        img_col = torch.zeros((N, C, kernel_height, kernel_width, out_height, out_width), device=input_data.device)

        for y in range(kernel_height):
            y_max = y + stride * out_height
            for x in range(kernel_width):
                x_max = x + stride * out_width
                img_col[:, :, y, x, :, :] = input_padded[:, :, y:y_max:stride, x:x_max:stride]

        return img_col.permute(0, 4, 5, 1, 2, 3).reshape(N * out_height * out_width, -1)

    def conv2d(self, input_signal, weights, stride, padding, groups, bias=None):
        """
        Performs a 2D convolution operation using the img2col method for optimization.

        This method applies a convolution operation to the input signal using the provided
        weights, stride, padding, and groups. It is optimized for SRAM by transforming
        the input signal using the img2col method and then performing matrix multiplication.

        Parameters:
        - input_signal (torch.Tensor): The input signal tensor of shape (N, C, H, W).
        - weights (torch.Tensor): The weight tensor for the convolution of shape (O, C, KH, KW).
        - stride (int): The stride of the convolution.
        - padding (int): The padding applied on both sides of the input signal.
        - groups (int): The number of groups to split the input signal and weights.
        - bias (torch.Tensor, optional): An optional bias tensor to add to the output.

        Returns:
        - torch.Tensor: The output tensor from the convolution operation.
        """
        batch_size, input_channels, H, W = input_signal.shape
        output_channels, k_channels, kernel_height, kernel_width = weights.shape
        group_input_channels = input_channels // groups
        group_output_channels = output_channels // groups

        new_height = (H + 2 * padding - kernel_height) // stride + 1
        new_width = (W + 2 * padding - kernel_width) // stride + 1

        input_pad = F.pad(input_signal, (padding, padding, padding, padding))
        output = torch.zeros((batch_size, output_channels, new_height, new_width), device=input_signal.device)

        for g in range(groups):
            group_input = input_pad[:, g * group_input_channels:(g + 1) * group_input_channels, :, :]
            group_weights = weights[g * group_output_channels:(g + 1) * group_output_channels, :, :, :]
            
            input_col = self.img2col(group_input, kernel_height, kernel_width, stride, padding)
            weight_col = group_weights.reshape(group_output_channels, -1)

            conv_out = torch.matmul(input_col, weight_col.T)
            conv_out = conv_out.reshape(batch_size, new_height, new_width, group_output_channels)
            output[:, g * group_output_channels:(g + 1) * group_output_channels, :, :] = conv_out.permute(0, 3, 1, 2)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

# Example Usage
batch_size, in_channels, height, width = 1, 3, 32, 32
out_channels, kernel_height, kernel_width = 8, 3, 3
input_data = torch.randint(0, 256, (batch_size, in_channels, height, width), dtype=torch.uint8)
weights = torch.randint(-128, 127, (out_channels, in_channels, kernel_height, kernel_width), dtype=torch.int8)

conv_layer = SRAMConv2d()
output = conv_layer.conv2d(input_data, weights, stride=1, padding=1, groups=1)
output_tensor = F.conv2d(input_data, weights, stride=1, padding=1, groups=1)
print('Output:', output)
print('torch-Output:', output_tensor)
# Testing against PyTorch Conv2D
tensor_all_true = torch.tensor(output, dtype=torch.int64) == output_tensor
print('Our Conv2D == Torch Conv2D?', tensor_all_true.all().item())
