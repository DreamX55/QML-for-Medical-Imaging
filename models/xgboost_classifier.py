"""
XGBoost Classifier for Baseline Comparison.

This module provides XGBoost-based classification on CNN features
as a classical ML baseline for comparison with the hybrid quantum model.

Design Decisions:
- Uses CNN features as input (not raw images)
- Provides feature importance extraction for interpretability
- Supports both CNN features and quantum features for analysis
- Includes hyperparameter configuration through config
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple
import torch
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, XGBoostConfig


class XGBoostClassifier:
    """
    XGBoost classifier wrapper for feature-based classification.
    
    This classifier works with pre-extracted CNN features,
    providing a classical ML baseline for comparison with
    the hybrid quantum-classical model.
    
    Attributes:
        config: XGBoost configuration
        model: XGBoost model instance
        label_encoder: Label encoder for class mapping
        feature_names: Names of input features
        num_classes: Number of output classes
    """
    
    def __init__(self, config: XGBoostConfig, num_classes: int):
        """
        Initialize XGBoost classifier.
        
        Args:
            config: XGBoost configuration object
            num_classes: Number of classes for classification
        """
        self.config = config
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
        
        # XGBoost parameters from config
        self.params = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'subsample': config.subsample,
            'colsample_bytree': config.colsample_bytree,
            'reg_alpha': config.reg_alpha,
            'reg_lambda': config.reg_lambda,
            'objective': config.objective,
            'random_state': config.seed,
            'n_jobs': config.n_jobs,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
        }
        
        print(f"XGBoostClassifier initialized with {num_classes} classes")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the XGBoost model.
        
        Args:
            X: Training features [n_samples, n_features]
            y: Training labels [n_samples]
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Names of features for interpretability
            early_stopping_rounds: Patience for early stopping
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Encode labels
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        
        # Create model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare evaluation set
        eval_set = [(X, y_encoded)]
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set.append((X_val, y_val_encoded))
        
        # Train
        self.model.fit(
            X, y_encoded,
            eval_set=eval_set,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        # Extract training history
        history = {
            'train_loss': self.model.evals_result()['validation_0']['mlogloss'],
        }
        if X_val is not None:
            history['val_loss'] = self.model.evals_result()['validation_1']['mlogloss']
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            Predicted labels [n_samples]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            Class probabilities [n_samples, num_classes]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Get importance scores
        importance = self.model.get_booster().get_score(
            importance_type=importance_type
        )
        
        # Map to feature names
        result = {}
        for i, name in enumerate(self.feature_names):
            key = f'f{i}'
            result[name] = importance.get(key, 0.0)
        
        # Normalize to sum to 1
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}
        
        return result
    
    def get_sorted_feature_importance(
        self,
        importance_type: str = 'gain',
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get feature importance sorted by score.
        
        Args:
            importance_type: Type of importance
            top_k: Return only top k features (None for all)
            
        Returns:
            List of (feature_name, importance) tuples, sorted descending
        """
        importance = self.get_feature_importance(importance_type)
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if top_k is not None:
            sorted_importance = sorted_importance[:top_k]
        
        return sorted_importance
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist(),
            'num_classes': self.num_classes,
        }
        
        metadata_path = Path(path).with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"XGBoost model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
        """
        # Create fresh model
        self.model = xgb.XGBClassifier(**self.params)
        self.model.load_model(path)
        
        # Load metadata
        metadata_path = Path(path).with_suffix('.meta.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_encoder.fit(metadata['classes'])
        self.num_classes = metadata['num_classes']
        self.is_fitted = True
        
        print(f"XGBoost model loaded from {path}")


def extract_features_from_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    feature_type: str = 'cnn'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from a hybrid model for XGBoost training.
    
    Args:
        model: HybridQuantumClassifier or similar
        dataloader: DataLoader with images
        device: Device to run inference on
        feature_type: 'cnn' or 'quantum'
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    model.eval()
    model.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            if feature_type == 'cnn':
                features = model.get_cnn_features(images)
            elif feature_type == 'quantum':
                features = model.get_quantum_features(images)
                if features is None:
                    raise ValueError("Model does not have quantum layer enabled")
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y


def compare_xgboost_on_features(
    cnn_features_train: np.ndarray,
    y_train: np.ndarray,
    cnn_features_test: np.ndarray,
    y_test: np.ndarray,
    quantum_features_train: Optional[np.ndarray] = None,
    quantum_features_test: Optional[np.ndarray] = None,
    config: Optional[XGBoostConfig] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare XGBoost performance on CNN vs quantum features.
    
    Args:
        cnn_features_train: CNN features for training
        y_train: Training labels
        cnn_features_test: CNN features for testing
        y_test: Test labels
        quantum_features_train: Quantum features for training (optional)
        quantum_features_test: Quantum features for testing (optional)
        config: XGBoost configuration
        
    Returns:
        Dictionary with performance metrics for each feature type
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    if config is None:
        from configs.config import XGBoostConfig
        config = XGBoostConfig()
    
    results = {}
    
    # Train on CNN features
    num_classes = len(np.unique(y_train))
    xgb_cnn = XGBoostClassifier(config, num_classes)
    xgb_cnn.fit(
        cnn_features_train, y_train,
        feature_names=[f'cnn_feat_{i}' for i in range(cnn_features_train.shape[1])],
        verbose=False
    )
    
    y_pred_cnn = xgb_cnn.predict(cnn_features_test)
    results['cnn_features'] = {
        'accuracy': accuracy_score(y_test, y_pred_cnn),
        'f1_macro': f1_score(y_test, y_pred_cnn, average='macro'),
        'feature_importance': xgb_cnn.get_sorted_feature_importance(top_k=5),
    }
    
    # Train on quantum features if provided
    if quantum_features_train is not None and quantum_features_test is not None:
        xgb_quantum = XGBoostClassifier(config, num_classes)
        xgb_quantum.fit(
            quantum_features_train, y_train,
            feature_names=[f'quantum_feat_{i}' for i in range(quantum_features_train.shape[1])],
            verbose=False
        )
        
        y_pred_quantum = xgb_quantum.predict(quantum_features_test)
        results['quantum_features'] = {
            'accuracy': accuracy_score(y_test, y_pred_quantum),
            'f1_macro': f1_score(y_test, y_pred_quantum, average='macro'),
            'feature_importance': xgb_quantum.get_sorted_feature_importance(top_k=5),
        }
    
    return results
