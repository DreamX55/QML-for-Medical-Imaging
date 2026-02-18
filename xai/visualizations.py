"""
XAI Visualization Utilities.

Provides publication-ready visualizations for:
- Grad-CAM heatmap overlays
- SHAP summary plots
- LIME explanations
- Combined explanation reports
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2


def plot_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM",
    alpha: float = 0.4,
    colormap: str = 'jet',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot Grad-CAM heatmap overlay on image.
    
    Args:
        image: Original image [H, W, C] in [0, 255] or [0, 1]
        heatmap: CAM heatmap [H, W] in [0, 1]
        title: Plot title
        alpha: Overlay transparency
        colormap: Matplotlib colormap
        save_path: Optional save path
        figsize: Figure size
    """
    # Normalize image to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap to match image
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Create colormap overlay
    heatmap_colored = plt.cm.get_cmap(colormap)(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create overlay
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap=colormap)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: List[str],
    X: Optional[np.ndarray] = None,
    max_display: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Plot SHAP summary as bar chart.
    
    Args:
        shap_values: SHAP values [n_samples, n_features]
        feature_names: Feature names
        X: Feature values for color coding
        max_display: Max features to show
        save_path: Optional save path
        figsize: Figure size
    """
    # Compute mean absolute SHAP values
    if isinstance(shap_values, list):
        # Multi-class: average across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)
    
    # Sort by importance
    indices = np.argsort(mean_abs)[::-1][:max_display]
    
    plt.figure(figsize=figsize)
    
    y_pos = np.arange(len(indices))
    plt.barh(y_pos, mean_abs[indices], color='steelblue', edgecolor='black')
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lime_explanation(
    image: np.ndarray,
    mask: np.ndarray,
    contributions: Optional[List] = None,
    title: str = "LIME Explanation",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot LIME explanation visualization.
    
    Args:
        image: Original image [H, W, C]
        mask: LIME mask [H, W]
        contributions: Optional superpixel contributions
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(image)
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')
    
    # Show positive/negative regions
    axes[1].imshow(mask, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title('LIME Regions', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy().astype(np.float32)
    pos_mask = (mask > 0).astype(np.float32)
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + pos_mask * 100, 0, 255)
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Important Regions', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_comparison(
    cnn_importance: Dict[str, float],
    quantum_importance: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Compare feature importance between CNN and quantum features.
    
    Args:
        cnn_importance: CNN feature importance dict
        quantum_importance: Quantum feature importance dict
        save_path: Optional save path
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2 if quantum_importance else 1, figsize=figsize)
    
    if not quantum_importance:
        axes = [axes]
    
    # CNN features
    names = list(cnn_importance.keys())
    values = list(cnn_importance.values())
    axes[0].barh(names, values, color='steelblue')
    axes[0].set_xlabel('Importance')
    axes[0].set_title('CNN Feature Importance', fontweight='bold')
    axes[0].invert_yaxis()
    
    # Quantum features
    if quantum_importance:
        names = list(quantum_importance.keys())
        values = list(quantum_importance.values())
        axes[1].barh(names, values, color='purple')
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Quantum Feature Importance', fontweight='bold')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_explanation_report(
    image: np.ndarray,
    gradcam_heatmap: np.ndarray,
    shap_importance: Dict[str, float],
    lime_mask: Optional[np.ndarray] = None,
    prediction: str = "",
    confidence: float = 0.0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> None:
    """
    Create comprehensive explanation report.
    
    Args:
        image: Original image
        gradcam_heatmap: Grad-CAM heatmap
        shap_importance: SHAP feature importance
        lime_mask: Optional LIME mask
        prediction: Predicted class name
        confidence: Prediction confidence
        save_path: Optional save path
        figsize: Figure size
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize heatmap
    if gradcam_heatmap.shape[:2] != image.shape[:2]:
        gradcam_heatmap = cv2.resize(gradcam_heatmap, (image.shape[1], image.shape[0]))
    
    fig = plt.figure(figsize=figsize)
    
    # Layout: 2 rows, 3 columns
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=11)
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(gradcam_heatmap, cmap='jet')
    ax2.set_title('Grad-CAM', fontsize=11)
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 3, 3)
    heatmap_colored = plt.cm.jet(gradcam_heatmap)[:, :, :3]
    overlay = 0.6 * image / 255.0 + 0.4 * heatmap_colored
    ax3.imshow(overlay)
    ax3.set_title('Overlay', fontsize=11)
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 3, 4)
    if lime_mask is not None:
        if lime_mask.shape[:2] != image.shape[:2]:
            lime_mask = cv2.resize(lime_mask.astype(np.float32), (image.shape[1], image.shape[0]))
        ax4.imshow(lime_mask, cmap='RdBu', vmin=-1, vmax=1)
        ax4.set_title('LIME Regions', fontsize=11)
    ax4.axis('off')
    
    ax5 = plt.subplot(2, 3, 5)
    names = list(shap_importance.keys())[:8]
    values = [shap_importance[n] for n in names]
    ax5.barh(names, values, color='steelblue')
    ax5.set_title('Feature Importance', fontsize=11)
    ax5.invert_yaxis()
    
    ax6 = plt.subplot(2, 3, 6)
    ax6.text(0.5, 0.6, f"Prediction: {prediction}", fontsize=14, 
             ha='center', va='center', fontweight='bold')
    ax6.text(0.5, 0.4, f"Confidence: {confidence:.1%}", fontsize=12,
             ha='center', va='center')
    ax6.axis('off')
    
    plt.suptitle('Model Explanation Report', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        normalize: Normalize by row
        save_path: Optional save path
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
