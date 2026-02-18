"""
Loss Functions for Medical Image Classification.

This module provides loss functions optimized for:
- Multi-class classification
- Class imbalanced datasets
- Label smoothing for regularization

Design Decisions:
- Cross-entropy as default for multi-class
- Label smoothing option for better generalization
- Focal loss option for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Label smoothing regularizes the model by replacing hard
    one-hot labels with soft labels, reducing overconfidence.
    
    For a target class y and smoothing factor ε:
    - Target class gets probability: 1 - ε + ε/K
    - Other classes get probability: ε/K
    where K is the number of classes.
    
    Attributes:
        smoothing: Smoothing factor ε ∈ [0, 1]
        num_classes: Number of classes K
    """
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 4):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing)
            num_classes: Number of classes
        """
        super().__init__()
        
        if not 0 <= smoothing < 1:
            raise ValueError(f"Smoothing must be in [0, 1), got {smoothing}")
        
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross-entropy loss.
        
        Args:
            predictions: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Scalar loss value
        """
        # Log-softmax for numerical stability
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Create one-hot targets
        one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        
        # Apply label smoothing
        smoothed_targets = one_hot * self.confidence + \
                          (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        
        # Compute loss
        loss = (-smoothed_targets * log_probs).sum(dim=-1)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard ones.
    Useful for imbalanced datasets where some classes dominate.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Attributes:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        num_classes: int = 4
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Class weights [num_classes] (None for uniform)
            gamma: Focusing parameter (0 = standard CE)
            num_classes: Number of classes
        """
        super().__init__()
        
        self.gamma = gamma
        self.num_classes = num_classes
        
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = alpha
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            predictions: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            Scalar loss value
        """
        # Move alpha to correct device
        if self.alpha.device != predictions.device:
            self.alpha = self.alpha.to(predictions.device)
        
        # Compute softmax probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Get probability of true class
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)
        
        # Get alpha for true class
        alpha_t = self.alpha[targets]
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute focal loss
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted cross-entropy loss for class imbalance.
    
    Simple alternative to focal loss that applies fixed
    weights to each class based on inverse frequency.
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        Initialize weighted CE loss.
        
        Args:
            class_weights: Weights for each class
        """
        super().__init__()
        self.class_weights = class_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy."""
        if self.class_weights is not None:
            weight = self.class_weights.to(predictions.device)
        else:
            weight = None
        
        return F.cross_entropy(predictions, targets, weight=weight)


def get_loss_function(
    num_classes: int,
    loss_type: str = 'cross_entropy',
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0
) -> nn.Module:
    """
    Factory function to get loss function.
    
    Args:
        num_classes: Number of classes
        loss_type: 'cross_entropy', 'focal', 'label_smoothing'
        label_smoothing: Smoothing factor for label smoothing
        class_weights: Optional class weights
        focal_gamma: Gamma for focal loss
        
    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                num_classes=num_classes
            )
        elif class_weights is not None:
            return WeightedCrossEntropyLoss(class_weights)
        else:
            return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            num_classes=num_classes
        )
    
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            smoothing=label_smoothing if label_smoothing > 0 else 0.1,
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
