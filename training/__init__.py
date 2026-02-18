"""
Training module for Hybrid Quantum-Classical ML.

Provides training loop, loss functions, metrics, and logging.
"""

from .trainer import Trainer, train_epoch, validate_epoch
from .losses import get_loss_function, LabelSmoothingCrossEntropy
from .metrics import compute_metrics, MetricsTracker
from .logger import TrainingLogger

__all__ = [
    "Trainer",
    "train_epoch",
    "validate_epoch",
    "get_loss_function",
    "LabelSmoothingCrossEntropy",
    "compute_metrics",
    "MetricsTracker",
    "TrainingLogger",
]
