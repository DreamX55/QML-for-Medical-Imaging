"""
Grad-CAM Implementation for CNN Visualization.

This module provides Gradient-weighted Class Activation Mapping
for visualizing which regions of the input image are important
for classification decisions.

Design Decisions:
- Hook-based implementation for flexibility
- Support for any target layer in the CNN
- Works with hybrid models (targets CNN component)
- Produces normalized heatmaps for overlay

Research Notes:
- Grad-CAM is effective for CNN interpretability
- For hybrid models, we analyze the classical CNN part
- Quantum layer contributions need different analysis methods
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from PIL import Image
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Creates visual explanations for CNN predictions by
    computing gradient-weighted combinations of feature maps.
    
    Algorithm:
    1. Forward pass to get feature maps at target layer
    2. Backward pass to get gradients w.r.t. target class
    3. Global average pool gradients to get channel weights
    4. Compute weighted sum of feature maps
    5. Apply ReLU and normalize
    
    Attributes:
        model: Model to explain (must have CNN component)
        target_layer: Layer to compute CAM for
        feature_maps: Stored feature maps (filled during forward)
        gradients: Stored gradients (filled during backward)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Model to analyze
            target_layer: Layer to compute CAM for (module reference)
            target_layer_name: Alternative: layer name string
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._get_layer_by_name(target_layer_name)
        else:
            # Default: last conv layer of CNN
            self.target_layer = self._find_last_conv_layer()
        
        # Storage for forward/backward hooks
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        print(f"GradCAM initialized for layer: {self.target_layer}")
    
    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get layer by dot-separated name."""
        parts = name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        
        def find_conv(module):
            nonlocal last_conv
            for child in module.children():
                if isinstance(child, nn.Conv2d):
                    last_conv = child
                find_conv(child)
        
        # Check if model has a CNN component
        if hasattr(self.model, 'cnn'):
            find_conv(self.model.cnn)
        else:
            find_conv(self.model)
        
        if last_conv is None:
            raise ValueError("Could not find convolutional layer in model")
        
        return last_conv
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image [1, C, H, W]
            target_class: Class to explain (None for predicted class)
            
        Returns:
            Heatmap array [H, W] with values in [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Enable gradients
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Compute weighted sum of feature maps
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        
        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Remove batch and channel dims
        cam = cam.squeeze(0).squeeze(0)  # [H, W]
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def get_cam_for_layer(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: nn.Module = None
    ) -> np.ndarray:
        """
        Compute CAM for a specific layer.
        
        Useful for analyzing different depth levels.
        """
        if target_layer is not None:
            # Temporarily change target layer
            original_layer = self.target_layer
            self.target_layer = target_layer
            self._register_hooks()
        
        cam = self(input_tensor, target_class)
        
        if target_layer is not None:
            self.target_layer = original_layer
            self._register_hooks()
        
        return cam


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ with improved weighting.
    
    Uses second-order gradients for better localization
    of fine-grained features.
    """
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Compute Grad-CAM++ heatmap."""
        # Forward pass
        self.model.eval()
        input_tensor.requires_grad = True
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Grad-CAM++ weighting
        gradients = self.gradients  # [1, C, H, W]
        feature_maps = self.feature_maps  # [1, C, H, W]
        
        # Second-order gradients
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3
        
        # Compute alpha weights
        spatial_sum = feature_maps.sum(dim=(2, 3), keepdim=True)
        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + spatial_sum * grad_3 + 1e-8
        alpha = alpha_numer / alpha_denom
        
        # Compute weights
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Weighted sum
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
    use_plus_plus: bool = False
) -> np.ndarray:
    """
    Convenience function to compute Grad-CAM.
    
    Args:
        model: Model to analyze
        input_tensor: Input image [1, C, H, W]
        target_class: Class to explain
        target_layer: Layer to compute CAM for
        use_plus_plus: Use Grad-CAM++ variant
        
    Returns:
        Heatmap array [H, W]
    """
    if use_plus_plus:
        cam = GradCAMPlusPlus(model, target_layer)
    else:
        cam = GradCAM(model, target_layer)
    
    return cam(input_tensor, target_class)


def resize_cam(cam: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize CAM heatmap to match input image size.
    
    Args:
        cam: Heatmap [H, W]
        target_size: (height, width) to resize to
        
    Returns:
        Resized heatmap
    """
    return cv2.resize(cam, (target_size[1], target_size[0]))


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay CAM heatmap on original image.
    
    Args:
        image: Original image [H, W, 3] in RGB, values in [0, 255]
        cam: CAM heatmap [H, W] in [0, 1]
        alpha: Transparency for heatmap
        colormap: OpenCV colormap
        
    Returns:
        Overlay image [H, W, 3]
    """
    # Resize CAM if needed
    if cam.shape[:2] != image.shape[:2]:
        cam = resize_cam(cam, image.shape[:2])
    
    # Convert CAM to colormap
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = (1 - alpha) * image + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay


def compute_gradcam_for_all_classes(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_classes: int,
    target_layer: Optional[nn.Module] = None
) -> List[np.ndarray]:
    """
    Compute Grad-CAM for all classes.
    
    Useful for comparing explanations across classes.
    
    Args:
        model: Model to analyze
        input_tensor: Input image [1, C, H, W]
        num_classes: Number of classes
        target_layer: Target layer
        
    Returns:
        List of heatmaps, one per class
    """
    cam = GradCAM(model, target_layer)
    
    cams = []
    for class_idx in range(num_classes):
        class_cam = cam(input_tensor.clone(), class_idx)
        cams.append(class_cam)
    
    return cams
