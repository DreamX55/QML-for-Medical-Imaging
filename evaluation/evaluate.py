"""
Model Evaluation Scripts.

Provides comprehensive evaluation including metrics,
confusion matrices, and per-class analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from training.metrics import (
    compute_metrics, compute_per_class_metrics,
    get_confusion_matrix, get_classification_report
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        dataloader: Test data loader
        device: Device to run on
        class_names: Optional class names
        
    Returns:
        Dictionary with all metrics and predictions
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probs)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    cm = get_confusion_matrix(y_true, y_pred)
    report = get_classification_report(y_true, y_pred, class_names, output_dict=True)
    
    return {
        'metrics': metrics,
        'per_class_metrics': per_class,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'labels': y_true.tolist(),
        'probabilities': y_proba.tolist(),
    }


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu',
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict:
    """
    Evaluate model on test set and optionally save results.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
        class_names: Class names
        save_path: Optional path to save results
        
    Returns:
        Evaluation results dictionary
    """
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print("="*50)
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")
    
    return results


def evaluate_quantum_contribution(
    hybrid_model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cpu'
) -> Dict:
    """
    Analyze quantum layer contribution to predictions.
    
    Args:
        hybrid_model: Hybrid model with quantum layer
        test_loader: Test data loader
        device: Device to run on
        
    Returns:
        Analysis results
    """
    hybrid_model.eval()
    hybrid_model.to(device)
    
    cnn_features_list = []
    quantum_features_list = []
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Get all features
            features = hybrid_model.get_all_features(images)
            
            cnn_features_list.append(features['cnn_features'].cpu().numpy())
            if 'quantum_features' in features:
                quantum_features_list.append(features['quantum_features'].cpu().numpy())
            
            preds = torch.argmax(features['logits'], dim=1)
            predictions_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.numpy())
    
    cnn_features = np.concatenate(cnn_features_list, axis=0)
    quantum_features = np.concatenate(quantum_features_list, axis=0) if quantum_features_list else None
    
    # Analyze feature statistics
    analysis = {
        'cnn_feature_mean': cnn_features.mean(axis=0).tolist(),
        'cnn_feature_std': cnn_features.std(axis=0).tolist(),
    }
    
    if quantum_features is not None:
        analysis['quantum_feature_mean'] = quantum_features.mean(axis=0).tolist()
        analysis['quantum_feature_std'] = quantum_features.std(axis=0).tolist()
        
        # Correlation between CNN and quantum features
        correlations = []
        for i in range(cnn_features.shape[1]):
            for j in range(quantum_features.shape[1]):
                corr = np.corrcoef(cnn_features[:, i], quantum_features[:, j])[0, 1]
                correlations.append(corr)
        analysis['cnn_quantum_correlation'] = np.mean(np.abs(correlations))
    
    return analysis
