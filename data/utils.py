"""
Data Utilities for Dataset Management.

This module provides utility functions for:
- Creating reproducible train/val/test splits
- Computing class weights for imbalanced data
- Dataset statistics computation
- Data export and visualization helpers

Design Decisions:
- Stratified splitting to maintain class distribution
- Support for reproducible random seeds
- Class weighting for handling imbalanced medical datasets
"""

import os
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def create_data_splits(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/validation/test splits with stratification.
    
    Stratification ensures each split has the same class distribution
    as the original dataset - critical for imbalanced medical data.
    
    Args:
        dataset: MRIDataset instance
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        stratify: Whether to stratify splits by class
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    n_samples = len(dataset)
    indices = list(range(n_samples))
    labels = [dataset.samples[i][1] for i in indices]
    
    # Set random seed
    np.random.seed(seed)
    
    if stratify:
        stratify_labels = labels
    else:
        stratify_labels = None
    
    # First split: separate test set
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_labels if stratify else None,
    )
    
    # Second split: separate train and validation
    # Adjust ratio for the remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify:
        train_val_labels = [labels[i] for i in train_val_indices]
    else:
        train_val_labels = None
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=train_val_labels,
    )
    
    # Print split statistics
    print(f"\nData Split Statistics:")
    print(f"  Train: {len(train_indices)} samples ({len(train_indices)/n_samples*100:.1f}%)")
    print(f"  Val:   {len(val_indices)} samples ({len(val_indices)/n_samples*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} samples ({len(test_indices)/n_samples*100:.1f}%)")
    
    # Verify stratification by checking class distribution
    if stratify:
        print("\nClass Distribution by Split:")
        for split_name, split_indices in [
            ('Train', train_indices),
            ('Val', val_indices),
            ('Test', test_indices)
        ]:
            split_labels = [labels[i] for i in split_indices]
            counter = Counter(split_labels)
            dist_str = ", ".join([
                f"C{k}: {v} ({v/len(split_indices)*100:.1f}%)" 
                for k, v in sorted(counter.items())
            ])
            print(f"  {split_name}: {dist_str}")
    
    return train_indices, val_indices, test_indices


def get_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = 'inverse'
) -> List[float]:
    """
    Compute class weights for handling imbalanced data.
    
    Methods:
    - 'inverse': Weight inversely proportional to class frequency
    - 'effective': Effective number of samples (better for long-tailed)
    - 'balanced': sklearn-style balanced weights
    
    Args:
        labels: List of class labels
        num_classes: Total number of classes
        method: Weighting method
        
    Returns:
        List of weights, one per class
    """
    counter = Counter(labels)
    n_samples = len(labels)
    
    if method == 'inverse':
        # Simple inverse frequency weighting
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 1)  # Avoid division by zero
            weights.append(n_samples / (num_classes * count))
    
    elif method == 'effective':
        # Effective number of samples (from "Class-Balanced Loss" paper)
        beta = 0.9999
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 1)
            effective_num = 1.0 - np.power(beta, count)
            weights.append((1.0 - beta) / effective_num)
    
    elif method == 'balanced':
        # sklearn-style balanced weights
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 1)
            weights.append(n_samples / count)
        # Normalize
        total = sum(weights)
        weights = [w / total * num_classes for w in weights]
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights


def compute_dataset_statistics(
    dataset,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Tuple[float, ...]]:
    """
    Compute per-channel mean and std of the dataset.
    
    Useful for computing custom normalization parameters
    instead of using ImageNet defaults.
    
    Args:
        dataset: MRIDataset instance
        sample_size: Number of samples to use (None for all)
        seed: Random seed for sampling
        
    Returns:
        Dictionary with 'mean' and 'std' tuples
    """
    from .transforms import get_val_transforms
    from configs.config import DataConfig
    
    # Use basic transforms without normalization
    basic_transform = torch.transforms.Compose([
        torch.transforms.Resize((224, 224)),
        torch.transforms.ToTensor(),
    ])
    
    # Sample indices
    n_samples = len(dataset)
    if sample_size is not None and sample_size < n_samples:
        np.random.seed(seed)
        indices = np.random.choice(n_samples, sample_size, replace=False)
    else:
        indices = range(n_samples)
    
    # Compute running statistics
    channels = 3  # Assuming RGB
    channel_sum = np.zeros(channels)
    channel_sum_sq = np.zeros(channels)
    n_pixels = 0
    
    print("Computing dataset statistics...")
    for idx in indices:
        img_path, _ = dataset.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img_tensor = basic_transform(img)
        
        # Update statistics
        channel_sum += img_tensor.sum(dim=[1, 2]).numpy()
        channel_sum_sq += (img_tensor ** 2).sum(dim=[1, 2]).numpy()
        n_pixels += img_tensor.shape[1] * img_tensor.shape[2]
    
    # Compute mean and std
    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)
    
    stats = {
        'mean': tuple(mean.tolist()),
        'std': tuple(std.tolist()),
    }
    
    print(f"Dataset Statistics:")
    print(f"  Mean: {stats['mean']}")
    print(f"  Std:  {stats['std']}")
    
    return stats


def save_split_indices(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    save_path: str
) -> None:
    """
    Save split indices to JSON for reproducibility.
    
    Args:
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        save_path: Path to save JSON file
    """
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    
    with open(save_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Split indices saved to {save_path}")


def load_split_indices(load_path: str) -> Tuple[List[int], List[int], List[int]]:
    """
    Load split indices from JSON.
    
    Args:
        load_path: Path to JSON file
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    with open(load_path, 'r') as f:
        splits = json.load(f)
    
    return splits['train'], splits['val'], splits['test']


def plot_class_distribution(
    dataset,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot the class distribution of the dataset.
    
    Args:
        dataset: MRIDataset instance
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    distribution = dataset.get_class_distribution()
    
    plt.figure(figsize=figsize)
    classes = list(distribution.keys())
    counts = list(distribution.values())
    
    colors = sns.color_palette("husl", len(classes))
    bars = plt.bar(classes, counts, color=colors, edgecolor='black')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Class distribution plot saved to {save_path}")
    
    plt.show()


def visualize_sample_images(
    dataset,
    num_samples: int = 12,
    cols: int = 4,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Visualize random sample images from the dataset.
    
    Args:
        dataset: MRIDataset instance
        num_samples: Number of samples to display
        cols: Number of columns in the grid
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    from PIL import Image
    
    n_samples = len(dataset)
    indices = np.random.choice(n_samples, min(num_samples, n_samples), replace=False)
    
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img_path, class_idx = dataset.samples[idx]
        class_name = dataset.idx_to_class[class_idx]
        
        img = Image.open(img_path).convert('RGB')
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name}\n{img_path.name}", fontsize=9)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")
    
    plt.show()
