"""Neural network architectures for Fashion MNIST classification."""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for Fashion MNIST.

    Architecture:
        Input (1, 28, 28)
        -> Conv2d(1, 32, 3, padding=1) -> [BatchNorm2d] -> ReLU -> MaxPool2d(2)
        -> Conv2d(32, 64, 3, padding=1) -> [BatchNorm2d] -> ReLU -> MaxPool2d(2)
        -> Flatten
        -> Linear(64*7*7, 128) -> ReLU -> [Dropout]
        -> Linear(128, 10)
        Output (10 classes)

    Args:
        use_batchnorm: Whether to use BatchNorm2d after conv layers
        dropout_rate: Dropout rate before final layer (0.0 = no dropout)
    """

    def __init__(self, use_batchnorm: bool = False, dropout_rate: float = 0.0):
        super(CNN, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # FC layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Fashion MNIST.

    A simple feedforward network that flattens the image and processes
    through fully connected layers. Used to demonstrate underfitting
    compared to CNN.

    Architecture:
        Input (28*28 = 784)
        -> Linear(784, 256) -> ReLU
        -> Linear(256, 128) -> ReLU
        -> Linear(128, 10)
        Output (10 classes)
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def get_model(model_type: str = "cnn", use_batchnorm: bool = False,
              dropout_rate: float = 0.0) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: "cnn" or "mlp"
        use_batchnorm: Whether to use BatchNorm (CNN only)
        dropout_rate: Dropout rate (CNN only)

    Returns:
        PyTorch model instance
    """
    if model_type.lower() == "cnn":
        return CNN(use_batchnorm=use_batchnorm, dropout_rate=dropout_rate)
    elif model_type.lower() == "mlp":
        return MLP()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'mlp'.")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
