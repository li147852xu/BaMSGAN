import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


# BasicBlock Class
class BasicBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and LeakyReLU activations.

    Args:
        in1 (int): Number of input channels.
    """
    def __init__(self, in1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in1, in1 * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in1 * 2, in1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass for the residual block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor after applying the residual block.
        """
        residual = x  # Store the original input
        out = self.conv1(x)  # First convolution
        out = self.relu1(out)  # First activation
        out = self.conv2(out)  # Second convolution
        out = self.relu2(out)  # Second activation
        out = residual + 0.1 * out  # Add the scaled residual
        return out


# SelfAttention Class
class SelfAttention(nn.Module):
    """
    Self-attention mechanism for enhancing feature representation.

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scalar for weighting the attention output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
        Forward pass for self-attention.

        Args:
            input (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor with enhanced features after self-attention.
        """
        batch_size, channels, height, width = input.shape

        # Query: reshape and permute to (batch_size, height*width, channels//8)
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)

        # Key: reshape to (batch_size, channels//8, height*width)
        k = self.key(input).view(batch_size, -1, height * width)

        # Value: reshape to (batch_size, channels, height*width)
        v = self.value(input).view(batch_size, -1, height * width)

        # Attention map: (batch_size, height*width, height*width)
        attn_matrix = torch.bmm(q, k)
        attn_matrix = self.softmax(attn_matrix)

        # Apply attention to the value tensor
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))
        out = out.view(*input.shape)  # Reshape back to input shape

        # Weighted sum of attention output and original input
        return self.gamma * out + input


# ConditionalBatchNorm2d Class
class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization for conditional GANs.

    Args:
        num_features (int): Number of features (channels) in the input tensor.
        num_classes (int): Number of conditional classes.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)  # Standard batch normalization without affine parameters
        self.embed = nn.Embedding(num_classes, num_features * 2)  # Embedding layer for class-specific parameters

        # Initialize embedding weights
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Scale (gamma) initialization
        self.embed.weight.data[:, num_features:].zero_()  # Shift (beta) initialization

    def forward(self, x, y):
        """
        Forward pass for conditional batch normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).
            y (Tensor): Class labels of shape (batch_size,).

        Returns:
            Tensor: Normalized and scaled tensor.
        """
        out = self.bn(x)  # Apply batch normalization
        gamma, beta = self.embed(y).chunk(2, 1)  # Split embedding into scale (gamma) and shift (beta)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)  # Apply scale and shift
        return out