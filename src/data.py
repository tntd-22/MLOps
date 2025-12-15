"""Data loading and augmentation for Fashion MNIST."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import DATA_DIR, DEFAULT_SUBSET_SIZE


def _get_stratified_indices(dataset, n_samples: int) -> list:
    """
    Get stratified sample indices ensuring balanced class distribution.

    Args:
        dataset: PyTorch dataset with targets attribute
        n_samples: Total number of samples to select

    Returns:
        List of indices with balanced class distribution
    """
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    n_classes = len(classes)
    samples_per_class = n_samples // n_classes

    indices = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        selected = np.random.choice(cls_indices, size=min(samples_per_class, len(cls_indices)), replace=False)
        indices.extend(selected.tolist())

    np.random.shuffle(indices)
    return indices


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
                    num_workers: int = 0, subset_size: int = DEFAULT_SUBSET_SIZE):
    """
    Create data loaders for Fashion MNIST.

    Args:
        batch_size: Batch size for training and validation
        use_augmentation: Whether to apply data augmentation
        num_workers: Number of worker processes for data loading
        subset_size: Number of training samples to use (None or 0 for full dataset)

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

    # Use subset for faster local training (stratified sampling for balanced classes)
    if subset_size and subset_size > 0:
        train_indices = _get_stratified_indices(train_dataset, subset_size)
        val_size = subset_size // 5  # ~1/5 ratio (5000 train → 1000 val)
        val_indices = _get_stratified_indices(val_dataset, val_size)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        print(f"Using subset: {len(train_indices)} train, {len(val_indices)} val samples (stratified)")

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
