"""
CNN Feature Extractor for Medical Image Classification.

This module provides a custom CNN architecture designed to:
1. Extract spatial features from MRI images
2. Compress features to ≤10 dimensions for quantum encoding
3. Be Grad-CAM compatible for explainability

Design Decisions:
- Custom lightweight CNN for full architectural control
- Global Average Pooling before dense layer (better than flatten for CAM)
- BatchNorm + Dropout for regularization
- Compression layer explicitly limited to 10 units
- Named layers for easy Grad-CAM targeting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import ModelConfig


class ConvBlock(nn.Module):
    """
    Convolutional block with optional batch normalization.
    
    Architecture: Conv2d -> BatchNorm (optional) -> ReLU -> MaxPool
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding
                bias=not use_batch_norm  # No bias if using BatchNorm
            )
        ]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        if pool_size > 1:
            layers.append(nn.MaxPool2d(pool_size))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNFeatureExtractor(nn.Module):
    """
    Custom CNN for extracting and compressing spatial features.
    
    Architecture:
    - Input: [B, C, H, W] image tensor
    - Multiple ConvBlocks with increasing channels
    - Global Average Pooling
    - Dense compression to num_features (≤10)
    - Output: [B, num_features] feature vector
    
    The last convolutional layer (`conv4`) is designed to be
    the target for Grad-CAM visualization.
    
    Attributes:
        features: Sequential container of conv blocks
        gap: Global Average Pooling layer
        compression: Dense layer for feature compression
        output_dim: Output feature dimension (≤10)
    """
    
    def __init__(self, config: ModelConfig, input_channels: int = 3):
        """
        Initialize the CNN feature extractor.
        
        Args:
            config: Model configuration object
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super().__init__()
        
        self.config = config
        self.output_dim = config.num_features
        
        # Validate output dimension
        if self.output_dim > 10:
            raise ValueError(
                f"Output dimension ({self.output_dim}) must be ≤10 for quantum encoding"
            )
        
        conv_channels = config.conv_channels
        kernel_size = config.kernel_size
        pool_size = config.pool_size
        use_batch_norm = config.use_batch_norm
        dropout_rate = config.dropout_rate
        
        # Build convolutional blocks in a single Sequential
        # Avoids duplicate state_dict keys from named attributes + sequential
        # features[3] (last block) is the target layer for Grad-CAM
        self.features = nn.Sequential(
            ConvBlock(input_channels, conv_channels[0],
                      kernel_size, pool_size, use_batch_norm, dropout_rate),
            ConvBlock(conv_channels[0], conv_channels[1],
                      kernel_size, pool_size, use_batch_norm, dropout_rate),
            ConvBlock(conv_channels[1], conv_channels[2],
                      kernel_size, pool_size, use_batch_norm, dropout_rate),
            ConvBlock(conv_channels[2], conv_channels[3],
                      kernel_size, pool_size, use_batch_norm, dropout_rate),
        )
        
        # Global Average Pooling
        # This aggregates spatial information and is CAM-friendly
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Compression layer
        # Reduces from conv_channels[-1] to num_features (≤10)
        self.compression = nn.Sequential(
            nn.Linear(conv_channels[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, self.output_dim),
            nn.BatchNorm1d(self.output_dim),  # Normalize to N(0,1) to avoid sigmoid saturation
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Feature vector [B, num_features]
        """
        # Extract spatial features
        features = self.features(x)
        
        # Global average pooling
        pooled = self.gap(features)
        
        # Flatten
        flattened = pooled.view(pooled.size(0), -1)
        
        # Compress to num_features
        compressed = self.compression(flattened)
        
        return compressed
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both feature maps (for Grad-CAM) and compressed features.
        
        This method is useful for XAI: it returns the Conv4 feature maps
        for Grad-CAM visualization along with the final compressed features.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Tuple of (feature_maps [B, C, H', W'], compressed_features [B, num_features])
        """
        # Extract features through conv layers
        feature_maps = self.features(x)
        
        # Continue through GAP and compression
        pooled = self.gap(feature_maps)
        flattened = pooled.view(pooled.size(0), -1)
        compressed = self.compression(flattened)
        
        return feature_maps, compressed
    
    def get_conv4_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get Conv4 feature maps only (for Grad-CAM).
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Feature maps from conv4 [B, C, H', W']
        """
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor using pretrained backbone.
    
    Uses ResNet18/34/50 with final layer replaced by compression layer.
    """
    
    def __init__(self, config: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.output_dim = config.num_features
        
        # Import torchvision
        from torchvision import models
        
        # Select backbone
        backbone_name = config.backbone.lower() if config.backbone else 'resnet18'
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            out_features = 512
        elif backbone_name == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            out_features = 512
        elif backbone_name == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            out_features = 2048
        else:
            # Fallback to defaults if unknown
            print(f"Warning: Unknown backbone {backbone_name}, using ResNet18")
            resnet = models.resnet18(pretrained=pretrained)
            out_features = 512
            
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Compression layer
        self.compression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, self.output_dim),
            nn.BatchNorm1d(self.output_dim),  # Normalize to N(0,1) for quantum encoding
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        compressed = self.compression(features)
        return compressed
        
    def freeze(self) -> None:
        """Freeze backbone weights (Quantum Transfer Learning)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("❄️  ResNet backbone frozen")
        
    def unfreeze(self) -> None:
        """Unfreeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 ResNet backbone unfrozen")


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-based feature extractor.
    
    Uses EfficientNet-B0/B1 etc.
    """
    
    def __init__(self, config: ModelConfig, pretrained: bool = True):
        super().__init__()
        self.config = config
        self.output_dim = config.num_features
        
        from torchvision import models
        
        # Map simple names to torchvision names
        name_map = {
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
        }
        
        backbone_name = config.backbone.lower()
        model_fn = name_map.get(backbone_name, models.efficientnet_b0)
        
        weights = 'DEFAULT' if pretrained else None
        efficientnet = model_fn(weights=weights)
        
        # Output features for B0 is 1280
        out_features = efficientnet.classifier[1].in_features
        
        # Remove classifier
        self.backbone = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Compression
        self.compression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, self.output_dim),
            nn.BatchNorm1d(self.output_dim),  # Normalize to N(0,1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.avgpool(x)
        return self.compression(x)
        
    def freeze(self) -> None:
        """Freeze backbone weights."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("❄️  EfficientNet backbone frozen")


def get_feature_extractor(config: ModelConfig, input_channels: int = 3) -> nn.Module:
    """Factory function."""
    if not config.backbone or config.backbone.lower() == 'custom':
        return CNNFeatureExtractor(config, input_channels)
    
    name = config.backbone.lower()
    if 'resnet' in name:
        return ResNetFeatureExtractor(config, pretrained=True)
    elif 'efficientnet' in name:
        return EfficientNetFeatureExtractor(config, pretrained=True)
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")
