"""
SHAP Explainer for Feature-based Explanations.

This module provides SHAP (SHapley Additive exPlanations) analysis
for understanding feature importance in the hybrid classifier.

SHAP values represent the contribution of each feature to the
prediction, based on game-theoretic Shapley values.

Design Decisions:
- DeepExplainer for neural networks (fast approximation)
- KernelExplainer as fallback for any model
- Support for both CNN features and quantum features
- Aggregated importance visualization
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import XAIConfig

# Import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based feature importance explainer.
    
    Supports multiple explanation modes:
    - 'deep': DeepExplainer for neural networks
    - 'kernel': KernelExplainer for any model
    - 'gradient': GradientExplainer for differentiable models
    
    Attributes:
        model: Model or feature extraction function
        explainer: SHAP explainer instance
        background_data: Background samples for explanation
        feature_names: Names of features being explained
    """
    
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        background_data: np.ndarray,
        mode: str = 'kernel',
        feature_names: Optional[List[str]] = None,
        config: Optional[XAIConfig] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Model or prediction function
            background_data: Background samples [n_samples, n_features]
            mode: Explainer mode ('deep', 'kernel', 'gradient')
            feature_names: Optional feature names
            config: XAI configuration
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.mode = mode
        self.background_data = background_data
        self.config = config or XAIConfig()
        
        # Set feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            n_features = background_data.shape[1]
            self.feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create explainer
        self._create_explainer()
        
        print(f"SHAP Explainer initialized with mode: {mode}")
        print(f"  Background samples: {len(background_data)}")
        print(f"  Features: {len(self.feature_names)}")
    
    def _create_explainer(self) -> None:
        """Create the appropriate SHAP explainer."""
        if self.mode == 'deep':
            # DeepExplainer for PyTorch models
            if isinstance(self.model, nn.Module):
                # Convert background to tensor
                background_tensor = torch.FloatTensor(self.background_data)
                self.explainer = shap.DeepExplainer(
                    self.model,
                    background_tensor
                )
            else:
                raise ValueError("DeepExplainer requires PyTorch model")
        
        elif self.mode == 'kernel':
            # KernelExplainer works with any function
            if isinstance(self.model, nn.Module):
                # Wrap model for numpy input
                def predict_fn(x):
                    with torch.no_grad():
                        x_tensor = torch.FloatTensor(x)
                        output = self.model(x_tensor)
                        return torch.softmax(output, dim=1).numpy()
                
                self.explainer = shap.KernelExplainer(
                    predict_fn,
                    self.background_data
                )
            else:
                self.explainer = shap.KernelExplainer(
                    self.model,
                    self.background_data
                )
        
        elif self.mode == 'gradient':
            # GradientExplainer for differentiable models
            if isinstance(self.model, nn.Module):
                background_tensor = torch.FloatTensor(self.background_data)
                self.explainer = shap.GradientExplainer(
                    self.model,
                    background_tensor
                )
            else:
                raise ValueError("GradientExplainer requires PyTorch model")
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def explain(
        self,
        X: np.ndarray,
        nsamples: int = 100
    ) -> np.ndarray:
        """
        Compute SHAP values for input samples.
        
        Args:
            X: Samples to explain [n_samples, n_features]
            nsamples: Number of background samples for kernel SHAP
            
        Returns:
            SHAP values [n_samples, n_features, n_classes]
        """
        if self.mode == 'kernel':
            shap_values = self.explainer.shap_values(X, nsamples=nsamples)
        else:
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            shap_values = self.explainer.shap_values(X)
        
        return shap_values
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        aggregation: str = 'mean_abs'
    ) -> Dict[str, float]:
        """
        Get aggregated feature importance from SHAP values.
        
        Args:
            X: Samples to explain
            aggregation: 'mean_abs', 'mean', or 'max'
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        shap_values = self.explain(X)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            # Average across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)
        
        # Aggregate across samples
        if aggregation == 'mean_abs':
            importance = np.mean(np.abs(shap_values), axis=0)
        elif aggregation == 'mean':
            importance = np.mean(shap_values, axis=0)
        elif aggregation == 'max':
            importance = np.max(np.abs(shap_values), axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Create dictionary
        importance_dict = {
            name: float(imp)
            for name, imp in zip(self.feature_names, importance)
        }
        
        return importance_dict
    
    def get_sorted_importance(
        self,
        X: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get feature importance sorted by magnitude.
        
        Args:
            X: Samples to explain
            top_k: Return only top k features
            
        Returns:
            Sorted list of (feature_name, importance) tuples
        """
        importance = self.get_feature_importance(X)
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            sorted_imp = sorted_imp[:top_k]
        
        return sorted_imp
    
    def explain_single(
        self,
        x: np.ndarray,
        class_idx: int = 0
    ) -> Dict[str, float]:
        """
        Explain a single sample for a specific class.
        
        Args:
            x: Single sample [n_features]
            class_idx: Class to explain
            
        Returns:
            Feature contributions for the sample
        """
        x = x.reshape(1, -1)
        shap_values = self.explain(x)
        
        if isinstance(shap_values, list):
            values = shap_values[class_idx][0]
        else:
            values = shap_values[0]
        
        return {
            name: float(val)
            for name, val in zip(self.feature_names, values)
        }


def explain_with_shap(
    model: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_background: int = 100,
    mode: str = 'kernel'
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convenience function for SHAP analysis.
    
    Args:
        model: Model to explain
        X_train: Training data for background
        X_test: Test data to explain
        feature_names: Feature names
        n_background: Number of background samples
        mode: SHAP explainer mode
        
    Returns:
        Tuple of (shap_values, importance_dict)
    """
    # Sample background data
    if len(X_train) > n_background:
        indices = np.random.choice(len(X_train), n_background, replace=False)
        background = X_train[indices]
    else:
        background = X_train
    
    # Create explainer
    explainer = SHAPExplainer(
        model,
        background,
        mode=mode,
        feature_names=feature_names
    )
    
    # Compute SHAP values
    shap_values = explainer.explain(X_test)
    importance = explainer.get_feature_importance(X_test)
    
    return shap_values, importance


class ImageSHAPExplainer:
    """
    SHAP explainer for image data using superpixel segmentation.
    
    Groups pixels into superpixels and computes SHAP values
    for each superpixel, showing which regions are important.
    """
    
    def __init__(
        self,
        model: nn.Module,
        preprocess_fn: Callable,
        n_segments: int = 50
    ):
        """
        Initialize image SHAP explainer.
        
        Args:
            model: Image classification model
            preprocess_fn: Preprocessing function for images
            n_segments: Number of superpixels
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required")
        
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.n_segments = n_segments
    
    def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment image into superpixels."""
        from skimage.segmentation import slic
        
        # SLIC segmentation
        segments = slic(
            image,
            n_segments=self.n_segments,
            compactness=10,
            sigma=1
        )
        
        return segments
    
    def _mask_image(
        self,
        image: np.ndarray,
        segments: np.ndarray,
        mask: np.ndarray,
        background: str = 'blur'
    ) -> np.ndarray:
        """Apply mask to image based on segments."""
        masked = image.copy()
        
        for i, include in enumerate(mask):
            if not include:
                if background == 'blur':
                    # Blur the masked region
                    import cv2
                    blur = cv2.GaussianBlur(image, (31, 31), 0)
                    masked[segments == i] = blur[segments == i]
                elif background == 'black':
                    masked[segments == i] = 0
                elif background == 'mean':
                    masked[segments == i] = image.mean()
        
        return masked
    
    def explain(
        self,
        image: np.ndarray,
        nsamples: int = 1000
    ) -> np.ndarray:
        """
        Compute SHAP values for image regions.
        
        Args:
            image: Input image [H, W, C]
            nsamples: Number of perturbation samples
            
        Returns:
            SHAP values for each superpixel
        """
        segments = self._segment_image(image)
        n_segments = len(np.unique(segments))
        
        # Create prediction function over masks
        def predict_fn(masks):
            outputs = []
            for mask in masks:
                masked_img = self._mask_image(image, segments, mask)
                input_tensor = self.preprocess_fn(masked_img)
                
                with torch.no_grad():
                    output = self.model(input_tensor.unsqueeze(0))
                    probs = torch.softmax(output, dim=1)
                outputs.append(probs.numpy()[0])
            
            return np.array(outputs)
        
        # Create explainer
        background = np.ones((1, n_segments))
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values
        test_mask = np.ones((1, n_segments))
        shap_values = explainer.shap_values(test_mask, nsamples=nsamples)
        
        return shap_values, segments


def create_shap_summary_plot(
    shap_values: np.ndarray,
    feature_names: List[str],
    X: np.ndarray,
    save_path: Optional[str] = None,
    max_display: int = 10
) -> None:
    """
    Create SHAP summary plot.
    
    Args:
        shap_values: SHAP values [n_samples, n_features]
        feature_names: Feature names
        X: Feature values for color coding
        save_path: Optional path to save plot
        max_display: Max features to show
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required")
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    
    plt.close()
