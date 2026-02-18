"""
MRI Dataset Class for Medical Image Classification.

This module provides a PyTorch Dataset implementation for loading
brain cancer MRI images organized in class-based folder structure.

Expected data structure:
    data_dir/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   └── ...
    └── class_n/
        └── ...

Design Decisions:
- Uses torchvision.datasets.ImageFolder internally for robust loading
- Supports custom transforms for train/val/test sets
- Maintains class-to-index mapping for interpretability
- Handles common MRI image formats (jpg, png, tiff)
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, DataConfig


class MRIDataset(Dataset):
    """
    Custom Dataset for Brain Cancer MRI Classification.
    
    Handles loading of MRI images from class-organized folders,
    applies preprocessing transforms, and provides class mappings.
    
    Attributes:
        root_dir: Path to the dataset root directory
        transform: Transforms to apply to images
        class_to_idx: Mapping from class names to indices
        idx_to_class: Reverse mapping from indices to class names
        samples: List of (image_path, class_idx) tuples
    """
    
    # Supported image extensions for MRI data
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Initialize the MRI Dataset.
        
        Args:
            root_dir: Path to directory containing class subfolders
            transform: Optional transforms to apply to images
            target_transform: Optional transforms to apply to labels
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # Validate directory exists
        if not self.root_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.root_dir}")
        
        # Discover classes from subdirectories
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {self.root_dir}")
        
        # Create class mappings
        self.class_to_idx: Dict[str, int] = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }
        self.idx_to_class: Dict[int, str] = {
            idx: cls_name for cls_name, idx in self.class_to_idx.items()
        }
        
        # Collect all samples
        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        for cls_name, cls_idx in self.class_to_idx.items():
            count = sum(1 for _, idx in self.samples if idx == cls_idx)
            print(f"  - {cls_name}: {count} images")
    
    def _load_samples(self) -> None:
        """Scan directories and collect all valid image samples."""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, class_index)
        """
        img_path, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Apply target transform if provided
        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)
        
        return image, class_idx
    
    def get_sample_path(self, idx: int) -> Path:
        """Get the file path for a sample (useful for debugging/XAI)."""
        return self.samples[idx][0]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of samples across classes."""
        distribution = {cls_name: 0 for cls_name in self.classes}
        for _, class_idx in self.samples:
            class_name = self.idx_to_class[class_idx]
            distribution[class_name] += 1
        return distribution


def get_dataloaders(
    config: Config,
    train_indices: Optional[List[int]] = None,
    val_indices: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, MRIDataset]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        config: Configuration object
        train_indices: Optional pre-computed train indices
        val_indices: Optional pre-computed validation indices
        test_indices: Optional pre-computed test indices
        device: Device string ('cuda', 'mps', 'cpu'). If None, auto-detected.
                Used to set pin_memory (True only for CUDA).
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, full_dataset)
    """
    import torch as _torch
    from .transforms import get_train_transforms, get_val_transforms
    from .utils import create_data_splits, get_class_weights
    
    data_config = config.data
    
    # Resolve device if not provided
    if device is None:
        if _torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # pin_memory=True only benefits CUDA (enables async DMA to GPU).
    # On MPS it triggers a "not supported" warning; on CPU it's a no-op.
    use_pin_memory = device.startswith('cuda')
    
    print(f"  Device: {device} | pin_memory: {use_pin_memory}")
    
    # Create base dataset without transforms for splitting
    full_dataset = MRIDataset(
        root_dir=data_config.data_dir,
        transform=None  # Will apply transforms to subsets
    )
    
    # Create or use provided splits
    if train_indices is None or val_indices is None or test_indices is None:
        train_indices, val_indices, test_indices = create_data_splits(
            dataset=full_dataset,
            train_ratio=data_config.train_ratio,
            val_ratio=data_config.val_ratio,
            test_ratio=data_config.test_ratio,
            seed=data_config.seed,
        )
    
    # Get transforms
    train_transform = get_train_transforms(data_config)
    val_transform = get_val_transforms(data_config)
    
    # Create subset datasets with appropriate transforms
    # We need wrapper datasets that apply transforms
    train_dataset = TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_dataset, val_indices, val_transform)
    test_dataset = TransformSubset(full_dataset, test_indices, val_transform)
    
    # Compute class weights for balanced sampling in training
    train_labels = [full_dataset.samples[i][1] for i in train_indices]
    class_weights = get_class_weights(train_labels, len(full_dataset.classes))
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        sampler=sampler,  # Use weighted sampler for class balance
        num_workers=data_config.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=use_pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=use_pin_memory,
    )
    
    return train_loader, val_loader, test_loader, full_dataset


class TransformSubset(Dataset):
    """
    A Subset that applies transforms to samples.
    
    This is necessary because we want different transforms for
    train/val/test splits but share the underlying dataset.
    """
    
    def __init__(
        self,
        dataset: MRIDataset,
        indices: List[int],
        transform: Optional[Callable] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        original_idx = self.indices[idx]
        img_path, class_idx = self.dataset.samples[original_idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_idx
    
    def get_sample_path(self, idx: int) -> Path:
        """Get the file path for a sample."""
        original_idx = self.indices[idx]
        return self.dataset.samples[original_idx][0]
