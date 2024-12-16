import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import numpy as np
from functools import reduce
from modules.layers import SelfAttention, BasicBlock
from modules.layers import ConditionalBatchNorm2d as CBN

# Constants
nz = 100  # Latent dimension
nc = 3    # Number of channels in the generated images
M = 3     # Number of some condition classes (if applicable)


# Generator Class
class Generator(nn.Module):
    """
    Generator for GAN. This class creates images from latent vectors.

    Args:
        image_size (int): The size of the generated image (default: 64).
        z_dim (int): Dimensionality of the latent vector (default: 100).
        conv_dim (int): Base convolution dimension for feature maps (default: 64).
    """
    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super().__init__()
        # Calculate the number of upsampling steps needed
        repeat_num = int(np.log2(image_size)) - 3
        mult = 2 ** repeat_num  # Multiplier for the first layer's channels

        # Initial layer for latent vector transformation
        self.l1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(z_dim, conv_dim * mult, kernel_size=4)),
            nn.LayerNorm([conv_dim * mult, 4, 4]),
            nn.ReLU()
        )

        # Upsampling layers
        curr_dim = conv_dim * mult
        self.l2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1)),
            nn.LayerNorm([curr_dim // 2, 8, 8]),
            nn.ReLU()
        )

        curr_dim //= 2
        self.l3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1)),
            nn.LayerNorm([curr_dim // 2, 16, 16]),
            nn.ReLU()
        )

        curr_dim //= 2
        self.l4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1)),
            nn.LayerNorm([curr_dim // 2, 32, 32]),
            nn.ReLU()
        )

        # Final layer to convert features into RGB image
        self.last = nn.Sequential(
            nn.ConvTranspose2d(curr_dim // 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Normalize outputs to [-1, 1]
        )

        # Self-attention layers for feature enhancement
        self.attn1 = SelfAttention(128)
        self.attn2 = SelfAttention(64)

    def forward(self, input):
        """
        Forward pass of the generator.

        Args:
            input (Tensor): Latent vector of shape (batch_size, z_dim).

        Returns:
            Tensor: Generated image of shape (batch_size, 3, image_size, image_size).
        """
        input = input.view(input.size(0), input.size(1), 1, 1)  # Reshape to (B, z_dim, 1, 1)
        out = self.l1(input)  # First layer
        out = self.l2(out)    # Second layer
        out = self.l3(out)    # Third layer
        out = self.attn1(out)  # Self-attention layer 1
        out = self.l4(out)    # Fourth layer
        out = self.attn2(out)  # Self-attention layer 2
        out = self.last(out)  # Final layer
        return out


# Discriminator Class
class Discriminator(nn.Module):
    """
    Discriminator for GAN. This class evaluates the realism of input images.

    Args:
        in_channels (int): Number of input image channels (default: 3).
        image_size (int): The size of input images (default: 256).
        ndf (int): Base convolution dimension for feature maps (default: 64).
    """
    def __init__(self, in_channels=3, image_size=256, ndf=64):
        super().__init__()

        # Helper function for convolutional layers
        def conv_2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.LeakyReLU(0.1)  # Leaky ReLU activation
            )

        # Convolutional blocks for feature extraction
        self.block_1 = conv_2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        current_dim = ndf
        self.block_2 = conv_2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1)
        current_dim *= 2
        self.block_3 = conv_2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1)
        current_dim *= 2

        # Self-attention layers
        self.attn_layer_1 = SelfAttention(current_dim)
        self.block_4 = conv_2d(current_dim, current_dim * 2, kernel_size=4, stride=2, padding=1)
        current_dim *= 2
        self.attn_layer_2 = SelfAttention(current_dim)

        # Final layer for classification
        self.last_layer = nn.Sequential(
            nn.Conv2d(current_dim, 1, kernel_size=4, stride=1)  # Outputs a single score
        )

    def forward(self, input):
        """
        Forward pass of the discriminator.

        Args:
            input (Tensor): Input image of shape (batch_size, in_channels, image_size, image_size).

        Returns:
            Tensor: Realism score for each image in the batch.
        """
        # Sequentially apply all layers using `reduce`
        all_layers = [self.block_1, self.block_2, self.block_3, self.attn_layer_1,
                      self.block_4, self.attn_layer_2, self.last_layer]
        out = reduce(lambda x, layer: layer(x), all_layers, input)
        return out