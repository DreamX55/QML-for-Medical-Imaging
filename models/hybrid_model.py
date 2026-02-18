"""
Hybrid Quantum-Classical Model for Medical Image Classification.

This module provides the main hybrid architecture that combines:
1. CNN feature extraction (spatial features)
2. Quantum layer (quantum-enhanced processing)
3. Classical classifier head (final predictions)

Design Decisions:
- End-to-end differentiable for joint optimization
- Optional bypass for classical-only comparison
- Modular structure for ablation studies
- Support for extracting intermediate features (for XAI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, ModelConfig, QuantumConfig
from .cnn_feature_extractor import CNNFeatureExtractor, get_feature_extractor
from quantum.quantum_layer import QuantumLayer


class HybridQuantumClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Classifier for Medical Images.
    
    Architecture:
    1. CNN Feature Extractor: MRI image → 10 compressed features
    2. Quantum Layer: 10 features → 10 quantum-enhanced features
    3. Classical Head: 10 features → num_classes logits
    
    The quantum layer can be bypassed for ablation studies.
    
    Attributes:
        cnn: CNN feature extractor
        quantum_layer: Parameterized quantum circuit layer
        classifier: Final classification head
        use_quantum: Whether to use quantum layer
    """
    
    def __init__(
        self,
        config: Config,
        use_quantum: bool = True,
        input_channels: int = 3
    ):
        """
        Initialize the hybrid classifier.
        
        Args:
            config: Full configuration object
            use_quantum: Whether to include quantum layer
            input_channels: Number of image input channels
        """
        super().__init__()
        
        self.config = config
        self.use_quantum = use_quantum
        self.num_classes = config.model.num_classes
        
        # CNN feature extractor
        self.cnn = get_feature_extractor(config.model, input_channels)
        
        # Quantum layer (optional)
        if use_quantum:
            self.quantum_layer = QuantumLayer(config.quantum)
            # Input to classifier comes from quantum layer
            classifier_input_dim = config.quantum.n_outputs
        else:
            self.quantum_layer = None
            classifier_input_dim = config.model.num_features
        
        # Classical post-processing head (REQUIRED after quantum layer)
        #
        # WHY quantum expectation values CANNOT be used directly as logits:
        #   1. Expectation values of Pauli-Z measurements are bounded to [-1, 1].
        #      Classification logits must be unbounded real numbers so that
        #      CrossEntropyLoss (which applies log-softmax internally) can
        #      produce well-calibrated gradients across the full range.
        #   2. With n_outputs == n_qubits (e.g., 10), the quantum output
        #      dimension doesn't match num_classes (e.g., 3). A learned
        #      linear projection is needed to map quantum features to class space.
        #   3. The bounded [-1, 1] range compresses gradient signal, making
        #      direct optimization unstable. A classical layer rescales and
        #      shifts the values into a useful logit range.
        #
        # This dense network transforms quantum features into raw logits:
        #   quantum_features [-1, 1] → Linear(n, 32) → ReLU → Dropout → Linear(32, num_classes) → logits (unbounded)
        #
        # NOTE: No softmax here. CrossEntropyLoss expects raw logits.
        # Softmax is applied only in predict_proba() for inference.
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(32, self.num_classes),
            # No softmax — CrossEntropyLoss handles it internally
        )
        
        # Store feature dimensions for XAI
        self.cnn_feature_dim = config.model.num_features
        self.quantum_feature_dim = config.quantum.n_outputs if use_quantum else 0
        
        print(f"\nHybridQuantumClassifier initialized:")
        print(f"  CNN output dim: {self.cnn_feature_dim}")
        print(f"  Quantum enabled: {use_quantum}")
        print(f"  Classifier input dim: {classifier_input_dim}")
        print(f"  Num classes: {self.num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Pipeline: CNN → feature compression → quantum layer → classical dense → logits
        
        Args:
            x: Input image tensor [batch, channels, height, width]
            
        Returns:
            Raw class logits [batch, num_classes] (unbounded, no softmax)
            Feed these directly to CrossEntropyLoss for training.
        """
        # Step 1: CNN feature extraction + compression
        # Output: [batch, num_features] (e.g., [batch, 10])
        cnn_features = self.cnn(x)
        
        # Step 2: Quantum processing (if enabled)
        if self.use_quantum:
            # Quantum layer: normalized features → PQC → expectation values
            # Output: [batch, n_outputs] with values in [-1, 1] (Pauli-Z expectations)
            # IMPORTANT: These are NOT logits — they must go through the classifier head
            quantum_features = self.quantum_layer(cnn_features)
            features = quantum_features
        else:
            features = cnn_features
        
        # Step 3: Classical post-processing → raw logits
        # Maps quantum features (bounded [-1,1]) to unbounded logit space
        # Output: [batch, num_classes] (e.g., [batch, 3])
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input image tensor
            
        Returns:
            Predicted class indices [batch]
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input image tensor
            
        Returns:
            Class probabilities [batch, num_classes]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features only (for XGBoost or XAI).
        
        Args:
            x: Input image tensor
            
        Returns:
            CNN features [batch, num_features]
        """
        return self.cnn(x)
    
    def get_quantum_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Extract quantum-enhanced features (for XAI).
        
        Args:
            x: Input image tensor
            
        Returns:
            Quantum features [batch, n_outputs] or None if quantum disabled
        """
        if not self.use_quantum:
            return None
        
        cnn_features = self.cnn(x)
        return self.quantum_layer(cnn_features)
    
    def get_all_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract all intermediate features for analysis.
        
        Args:
            x: Input image tensor
            
        Returns:
            Dictionary with 'cnn_features', 'quantum_features', 'logits'
        """
        cnn_features = self.cnn(x)
        
        result = {'cnn_features': cnn_features}
        
        if self.use_quantum:
            quantum_features = self.quantum_layer(cnn_features)
            result['quantum_features'] = quantum_features
            logits = self.classifier(quantum_features)
        else:
            logits = self.classifier(cnn_features)
        
        result['logits'] = logits
        
        return result
    
    def get_feature_maps_for_gradcam(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get CNN feature maps for Grad-CAM visualization.
        
        Args:
            x: Input image tensor
            
        Returns:
            Tuple of (feature_maps, predictions)
        """
        feature_maps, cnn_features = self.cnn.get_feature_maps(x)
        
        if self.use_quantum:
            quantum_features = self.quantum_layer(cnn_features)
            logits = self.classifier(quantum_features)
        else:
            logits = self.classifier(cnn_features)
        
        return feature_maps, logits


class ClassicalOnlyClassifier(nn.Module):
    """
    Classical-only baseline for comparison.
    
    Same architecture as HybridQuantumClassifier but without
    the quantum layer. Useful for ablation studies.
    """
    
    def __init__(self, config: Config, input_channels: int = 3):
        """Initialize classical classifier."""
        super().__init__()
        
        self.config = config
        self.num_classes = config.model.num_classes
        
        # CNN feature extractor
        self.cnn = get_feature_extractor(config.model, input_channels)
        
        # Classification head (same as hybrid but without quantum)
        self.classifier = nn.Sequential(
            nn.Linear(config.model.num_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.model.dropout_rate),
            nn.Linear(32, self.num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.cnn(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get CNN features."""
        return self.cnn(x)


class EnsembleClassifier(nn.Module):
    """
    Ensemble of hybrid and classical classifiers.
    
    Combines predictions from hybrid quantum-classical model
    and classical-only model for potentially improved performance.
    """
    
    def __init__(
        self,
        config: Config,
        input_channels: int = 3,
        ensemble_weights: Tuple[float, float] = (0.6, 0.4)
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            config: Configuration object
            input_channels: Number of input channels
            ensemble_weights: Weights for (hybrid, classical) predictions
        """
        super().__init__()
        
        self.hybrid = HybridQuantumClassifier(config, use_quantum=True, input_channels=input_channels)
        self.classical = ClassicalOnlyClassifier(config, input_channels)
        self.ensemble_weights = ensemble_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble averaging."""
        hybrid_logits = self.hybrid(x)
        classical_logits = self.classical(x)
        
        # Weighted average of logits
        logits = (
            self.ensemble_weights[0] * hybrid_logits +
            self.ensemble_weights[1] * classical_logits
        )
        
        return logits


def create_model(
    config: Config,
    model_type: str = 'hybrid',
    input_channels: int = 3
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        config: Configuration object
        model_type: 'hybrid', 'classical', or 'ensemble'
        input_channels: Number of input channels
        
    Returns:
        Model instance
    """
    if model_type == 'hybrid':
        return HybridQuantumClassifier(config, use_quantum=True, input_channels=input_channels)
    elif model_type == 'classical':
        return HybridQuantumClassifier(config, use_quantum=False, input_channels=input_channels)
    elif model_type == 'ensemble':
        return EnsembleClassifier(config, input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by component
    counts = {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }
    
    if hasattr(model, 'cnn'):
        cnn_params = sum(p.numel() for p in model.cnn.parameters())
        counts['cnn'] = cnn_params
    
    if hasattr(model, 'quantum_layer') and model.quantum_layer is not None:
        quantum_params = sum(p.numel() for p in model.quantum_layer.parameters())
        counts['quantum'] = quantum_params
    
    if hasattr(model, 'classifier'):
        classifier_params = sum(p.numel() for p in model.classifier.parameters())
        counts['classifier'] = classifier_params
    
    return counts
