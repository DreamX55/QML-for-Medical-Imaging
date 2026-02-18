"""
Metrics for Model Evaluation.

This module provides comprehensive metrics for medical image
classification including:
- Accuracy, Precision, Recall, F1
- AUC-ROC for multi-class
- Confusion matrix
- Per-class metrics

Design Decisions:
- Support for both single evaluations and tracking over epochs
- Multi-class AUC using one-vs-rest
- Detailed per-class breakdowns for medical applications
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels [n_samples]
        y_pred: Predicted labels [n_samples]
        y_proba: Predicted probabilities [n_samples, n_classes] (for AUC)
        average: Averaging method for multi-class ('macro', 'micro', 'weighted')
        
    Returns:
        Dictionary of metric names to values
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    # Compute AUC if probabilities provided
    if y_proba is not None:
        try:
            # Multi-class AUC using one-vs-rest
            n_classes = y_proba.shape[1]
            if n_classes == 2:
                # Binary case
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class case
                auc = roc_auc_score(
                    y_true, 
                    y_proba, 
                    multi_class='ovr',
                    average=average
                )
            metrics['auc'] = auc
        except ValueError:
            # AUC undefined if not all classes present in y_true
            metrics['auc'] = 0.0
    
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class separately.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary mapping class name to metrics dict
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes)]
    
    per_class = {}
    
    for i, class_name in enumerate(class_names):
        if i >= n_classes:
            break
            
        # Binary metrics for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        per_class[class_name] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'support': int(y_true_binary.sum()),  # Number of true samples
        }
    
    return per_class


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        normalize: Whether to normalize by row (true labels)
        
    Returns:
        Confusion matrix [n_classes, n_classes]
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        # Normalize by row (recall normalization)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype(float) / row_sums
    
    return cm


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> str:
    """
    Get sklearn classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional class names
        output_dict: Return as dictionary instead of string
        
    Returns:
        Classification report as string or dict
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )


class MetricsTracker:
    """
    Track metrics over training epochs.
    
    Provides convenience methods for:
    - Storing metrics per epoch
    - Computing running averages
    - Getting best metrics
    - Plotting metric curves
    """
    
    def __init__(self, metric_names: List[str] = None):
        """
        Initialize metrics tracker.
        
        Args:
            metric_names: Names of metrics to track
        """
        if metric_names is None:
            metric_names = ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        
        self.metric_names = metric_names
        self.history = {name: {'train': [], 'val': []} for name in metric_names}
        self.best_metrics = {}
    
    def update(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """
        Update with metrics from one epoch.
        
        Args:
            train_metrics: Training metrics dict
            val_metrics: Validation metrics dict
        """
        for name in self.metric_names:
            if name in train_metrics:
                self.history[name]['train'].append(train_metrics[name])
            if name in val_metrics:
                self.history[name]['val'].append(val_metrics[name])
        
        # Update best metrics
        if 'accuracy' in val_metrics:
            if 'val_accuracy' not in self.best_metrics or \
               val_metrics['accuracy'] > self.best_metrics['val_accuracy']:
                self.best_metrics['val_accuracy'] = val_metrics['accuracy']
                self.best_metrics['best_epoch'] = len(self.history['accuracy']['val'])
        
        if 'loss' in val_metrics:
            if 'val_loss' not in self.best_metrics or \
               val_metrics['loss'] < self.best_metrics['val_loss']:
                self.best_metrics['val_loss'] = val_metrics['loss']
    
    def get_last(self, metric_name: str, split: str = 'val') -> float:
        """Get the last recorded value for a metric."""
        if metric_name in self.history and self.history[metric_name][split]:
            return self.history[metric_name][split][-1]
        return 0.0
    
    def get_best(self, metric_name: str = 'accuracy', split: str = 'val') -> Tuple[float, int]:
        """
        Get the best value and epoch for a metric.
        
        Args:
            metric_name: Metric to check
            split: 'train' or 'val'
            
        Returns:
            Tuple of (best_value, best_epoch)
        """
        values = self.history.get(metric_name, {}).get(split, [])
        if not values:
            return 0.0, 0
        
        if metric_name == 'loss':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return values[best_idx], best_idx + 1
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of best metrics."""
        summary = {}
        
        for name in self.metric_names:
            if name == 'loss':
                best_val, epoch = self.get_best(name, 'val')
                summary[f'best_val_{name}'] = best_val
            else:
                best_val, epoch = self.get_best(name, 'val')
                summary[f'best_val_{name}'] = best_val
        
        return summary
    
    def to_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        
        data = {}
        for name in self.metric_names:
            if self.history[name]['train']:
                data[f'train_{name}'] = self.history[name]['train']
            if self.history[name]['val']:
                data[f'val_{name}'] = self.history[name]['val']
        
        return pd.DataFrame(data)


def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int = 1
) -> Tuple[float, float]:
    """
    Compute sensitivity (recall) and specificity.
    
    Important for medical applications where both
    false positives and false negatives matter.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_idx: Index of positive class
        
    Returns:
        Tuple of (sensitivity, specificity)
    """
    # Binary conversion
    y_true_binary = (y_true == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    # True positives, negatives, etc.
    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    
    # Sensitivity = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity
