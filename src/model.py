"""Neural network architectures for Fashion MNIST classification."""

from typing import List
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Reusable convolutional block: Conv2d → [BatchNorm2d] → ReLU → MaxPool2d

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for convolution
        use_batchnorm: Whether to use BatchNorm2d after convolution
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batchnorm: bool = False
    ):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNN(nn.Module):
    """
    Configurable Convolutional Neural Network for Fashion MNIST.

    Architecture is determined by the channels list:
        - channels=[8] → 1 conv block (tiny, for underfitting demo)
        - channels=[32, 64] → 2 conv blocks (standard)

    Each conv block halves spatial dimensions via MaxPool2d(2).
    Input: (1, 28, 28) → after n blocks: (channels[-1], 28/2^n, 28/2^n)

    Args:
        channels: List of output channels for each conv block (e.g., [8] or [32, 64])
        kernel_size: Kernel size for all conv layers
        use_batchnorm: Whether to use BatchNorm2d after conv layers
        dropout_rate: Dropout rate before final layer (0.0 = no dropout)
    """

    def __init__(
        self,
        channels: List[int] = None,
        kernel_size: int = 3,
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0
    ):
        super(CNN, self).__init__()

        if channels is None:
            channels = [32, 64]

        self.channels = channels
        self.kernel_size = kernel_size
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        # Build conv blocks dynamically
        in_channels = 1  # Fashion MNIST is grayscale
        self.conv_blocks = nn.ModuleList()
        for out_channels in channels:
            self.conv_blocks.append(
                ConvBlock(in_channels, out_channels, kernel_size, use_batchnorm)
            )
            in_channels = out_channels

        # Calculate flattened size: 28 / (2^num_blocks) for each spatial dim
        num_blocks = len(channels)
        spatial_size = 28 // (2 ** num_blocks)
        flatten_size = channels[-1] * spatial_size * spatial_size

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Pass through all conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # FC layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_model(
    channels: List[int] = None,
    kernel_size: int = 3,
    use_batchnorm: bool = False,
    dropout_rate: float = 0.0
) -> nn.Module:
    """
    Factory function to create CNN models.

    Args:
        channels: List of output channels for each conv block (e.g., [8] or [32, 64])
        kernel_size: Kernel size for all conv layers
        use_batchnorm: Whether to use BatchNorm
        dropout_rate: Dropout rate

    Returns:
        PyTorch CNN model instance
    """
    return CNN(
        channels=channels,
        kernel_size=kernel_size,
        use_batchnorm=use_batchnorm,
        dropout_rate=dropout_rate
    )


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
