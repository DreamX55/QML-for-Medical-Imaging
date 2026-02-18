"""
Models module for Hybrid Quantum-Classical ML.

Provides CNN feature extraction, quantum layer integration,
hybrid model architecture, and XGBoost baseline.
"""

from .cnn_feature_extractor import CNNFeatureExtractor
from .hybrid_model import HybridQuantumClassifier
from .xgboost_classifier import XGBoostClassifier

__all__ = [
    "CNNFeatureExtractor",
    "HybridQuantumClassifier",
    "XGBoostClassifier",
]
