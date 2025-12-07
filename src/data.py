"""Data loading and augmentation for Fashion MNIST."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import DATA_DIR


def get_transforms(use_augmentation: bool = False):
    """
    Get data transforms for training and validation.

    Args:
        use_augmentation: Whether to apply data augmentation to training data

    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Validation transform (always the same)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if use_augmentation:
        # Training transform with augmentation
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        # Training transform without augmentation
        train_transform = val_transform

    return train_transform, val_transform


def get_dataloaders(batch_size: int = 64, use_augmentation: bool = False,
                    num_workers: int = 0):
    """
    Create data loaders for Fashion MNIST.

    Args:
        batch_size: Batch size for training and validation
        use_augmentation: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_transform, val_transform = get_transforms(use_augmentation)

    # Download and load training data
    train_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=train_transform
    )

    # Download and load validation data
    val_dataset = datasets.FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


def get_inference_transform():
    """Get transform for inference (same as validation)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
