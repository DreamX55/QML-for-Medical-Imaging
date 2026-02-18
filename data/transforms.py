"""
Image Transforms for MRI Preprocessing.

This module provides transform pipelines for training, validation, and test sets.
Transforms handle resizing, normalization, and data augmentation.

Design Decisions:
- Use torchvision.transforms for consistency with PyTorch ecosystem
- Training transforms include augmentation for generalization
- Validation/test transforms are deterministic for reproducibility
- Normalization uses configurable mean/std (default ImageNet, adjust for MRI)
"""

from typing import Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import DataConfig


def get_train_transforms(config: DataConfig) -> transforms.Compose:
    """
    Get transforms for training data.
    
    Includes data augmentation to improve generalization:
    - Random rotation
    - Random horizontal flip
    - Color jitter (subtle, as MRI intensity matters)
    - Random affine transformations
    
    Args:
        config: Data configuration object
        
    Returns:
        Composed transform pipeline
    """
    transform_list = [
        # Resize to target size
        transforms.Resize(config.image_size),
    ]
    
    if config.use_augmentation:
        # Random rotation
        if config.random_rotation_degrees > 0:
            transform_list.append(
                transforms.RandomRotation(config.random_rotation_degrees)
            )
        
        # Random horizontal flip
        if config.random_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Random vertical flip (less common for MRI)
        if config.random_vertical_flip:
            transform_list.append(transforms.RandomVerticalFlip(p=0.5))
        
        # Subtle random affine for scale/translation invariance
        transform_list.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            )
        )
        
        # Very subtle color jitter - MRI intensity is important
        # So we only apply minimal brightness/contrast changes
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            )
        )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        ),
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: DataConfig) -> transforms.Compose:
    """
    Get transforms for validation data.
    
    No augmentation - deterministic for consistent evaluation.
    
    Args:
        config: Data configuration object
        
    Returns:
        Composed transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        ),
    ])


def get_test_transforms(config: DataConfig) -> transforms.Compose:
    """
    Get transforms for test data.
    
    Same as validation - deterministic for reproducible evaluation.
    
    Args:
        config: Data configuration object
        
    Returns:
        Composed transform pipeline
    """
    return get_val_transforms(config)


class MRINoiseReduction:
    """
    Custom transform for MRI-specific noise reduction.
    
    Uses median filtering to reduce salt-and-pepper noise
    common in MRI scans while preserving edges.
    
    This is an optional preprocessing step that can be added
    to the transform pipeline if noise is a concern.
    """
    
    def __init__(self, kernel_size: int = 3):
        """
        Initialize noise reduction transform.
        
        Args:
            kernel_size: Size of the median filter kernel (must be odd)
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        self.kernel_size = kernel_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply median filter to reduce noise."""
        import cv2
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Apply median filter
        if len(img_array.shape) == 3:
            # Color image - apply to each channel
            filtered = np.stack([
                cv2.medianBlur(img_array[:, :, c], self.kernel_size)
                for c in range(img_array.shape[2])
            ], axis=2)
        else:
            # Grayscale
            filtered = cv2.medianBlur(img_array, self.kernel_size)
        
        return Image.fromarray(filtered)


class CLAHEEnhancement:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Enhances local contrast in MRI images, useful for improving
    visibility of subtle features in low-contrast regions.
    
    Research note: CLAHE is commonly used in medical imaging
    to improve visualization without over-amplifying noise.
    """
    
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize CLAHE enhancement.
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE enhancement."""
        import cv2
        
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )
        
        if len(img_array.shape) == 3:
            # Convert to LAB color space for CLAHE on luminance channel
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale - apply directly
            enhanced = clahe.apply(img_array)
        
        return Image.fromarray(enhanced)


def get_mri_enhanced_transforms(config: DataConfig, training: bool = True) -> transforms.Compose:
    """
    Get enhanced transform pipeline with MRI-specific preprocessing.
    
    This is an alternative pipeline that includes:
    - Noise reduction
    - CLAHE contrast enhancement
    - Standard augmentation (for training)
    
    Use this for challenging datasets with noise or low contrast.
    
    Args:
        config: Data configuration object
        training: Whether to include augmentation
        
    Returns:
        Composed transform pipeline
    """
    transform_list = [
        # MRI-specific preprocessing
        MRINoiseReduction(kernel_size=3),
        CLAHEEnhancement(clip_limit=2.0),
        
        # Standard resize
        transforms.Resize(config.image_size),
    ]
    
    if training and config.use_augmentation:
        transform_list.extend([
            transforms.RandomRotation(config.random_rotation_degrees),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        ),
    ])
    
    return transforms.Compose(transform_list)


def inverse_normalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """
    Inverse normalization to recover original image values.
    
    Useful for visualization and XAI overlay purposes.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean (same as used in transforms)
        std: Normalization std (same as used in transforms)
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.device.type != 'cpu':
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    if tensor.dim() == 4:
        # Batch dimension present
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)
