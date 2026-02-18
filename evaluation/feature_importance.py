"""
Feature Importance Analysis.

Analyzes feature importance from XGBoost and compares
CNN features vs quantum-enhanced features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.xgboost_classifier import XGBoostClassifier, extract_features_from_model


def analyze_feature_importance(
    hybrid_model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: str = 'cpu',
    save_dir: str = './outputs/feature_analysis'
) -> Dict:
    """
    Analyze feature importance using XGBoost.
    
    Args:
        hybrid_model: Trained hybrid model
        train_loader: Training data
        test_loader: Test data
        config: Configuration
        device: Device to use
        save_dir: Directory to save results
        
    Returns:
        Feature importance analysis results
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    hybrid_model.eval()
    
    # Extract CNN features
    print("Extracting CNN features...")
    cnn_train, y_train = extract_features_from_model(
        hybrid_model, train_loader, device, feature_type='cnn'
    )
    cnn_test, y_test = extract_features_from_model(
        hybrid_model, test_loader, device, feature_type='cnn'
    )
    
    # Extract quantum features if available
    quantum_train, quantum_test = None, None
    try:
        quantum_train, _ = extract_features_from_model(
            hybrid_model, train_loader, device, feature_type='quantum'
        )
        quantum_test, _ = extract_features_from_model(
            hybrid_model, test_loader, device, feature_type='quantum'
        )
        print("Quantum features extracted.")
    except ValueError:
        print("No quantum features available.")
    
    results = {}
    
    # XGBoost on CNN features
    print("\nTraining XGBoost on CNN features...")
    n_classes = len(np.unique(y_train))
    xgb_cnn = XGBoostClassifier(config.xgboost, n_classes)
    feature_names = [f'cnn_{i}' for i in range(cnn_train.shape[1])]
    xgb_cnn.fit(cnn_train, y_train, cnn_test, y_test, feature_names, verbose=False)
    
    cnn_importance = xgb_cnn.get_sorted_feature_importance()
    results['cnn_feature_importance'] = cnn_importance
    
    # XGBoost on quantum features
    if quantum_train is not None:
        print("Training XGBoost on quantum features...")
        xgb_quantum = XGBoostClassifier(config.xgboost, n_classes)
        q_names = [f'quantum_{i}' for i in range(quantum_train.shape[1])]
        xgb_quantum.fit(quantum_train, y_train, quantum_test, y_test, q_names, verbose=False)
        
        quantum_importance = xgb_quantum.get_sorted_feature_importance()
        results['quantum_feature_importance'] = quantum_importance
    
    # Visualize
    _plot_feature_importance(cnn_importance, "CNN", f'{save_dir}/cnn_importance.png')
    if quantum_train is not None:
        _plot_feature_importance(quantum_importance, "Quantum", f'{save_dir}/quantum_importance.png')
    
    return results


def _plot_feature_importance(
    importance: List[Tuple[str, float]],
    title: str,
    save_path: str
) -> None:
    """Plot feature importance bar chart."""
    names = [x[0] for x in importance]
    values = [x[1] for x in importance]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), values, color='steelblue')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance')
    plt.title(f'{title} Feature Importance (XGBoost)', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def compare_feature_spaces(
    cnn_features: np.ndarray,
    quantum_features: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> Dict:
    """
    Compare CNN and quantum feature spaces.
    
    Args:
        cnn_features: CNN features [n_samples, n_cnn]
        quantum_features: Quantum features [n_samples, n_quantum]
        labels: Class labels
        save_path: Optional save path
        
    Returns:
        Comparison statistics
    """
    from sklearn.manifold import TSNE
    
    # t-SNE visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # CNN t-SNE
    tsne_cnn = TSNE(n_components=2, random_state=42)
    cnn_2d = tsne_cnn.fit_transform(cnn_features)
    scatter1 = axes[0].scatter(cnn_2d[:, 0], cnn_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    axes[0].set_title('CNN Features (t-SNE)', fontsize=12)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Quantum t-SNE
    tsne_q = TSNE(n_components=2, random_state=42)
    q_2d = tsne_q.fit_transform(quantum_features)
    scatter2 = axes[1].scatter(q_2d[:, 0], q_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    axes[1].set_title('Quantum Features (t-SNE)', fontsize=12)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    
    plt.colorbar(scatter2, ax=axes[1], label='Class')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Compute separability metrics
    from sklearn.metrics import silhouette_score
    
    stats = {
        'cnn_silhouette': silhouette_score(cnn_features, labels),
        'quantum_silhouette': silhouette_score(quantum_features, labels),
    }
    
    return stats
