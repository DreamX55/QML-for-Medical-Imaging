"""
LIME Explainer for Local Interpretable Explanations.

Provides LIME (Local Interpretable Model-agnostic Explanations)
for understanding individual predictions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple
import warnings

try:
    import lime
    import lime.lime_image
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class LIMEExplainer:
    """LIME-based local explanation for predictions."""
    
    def __init__(
        self,
        model: nn.Module,
        mode: str = 'image',
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
        training_data: Optional[np.ndarray] = None
    ):
        if not LIME_AVAILABLE:
            raise ImportError("LIME required: pip install lime")
        
        self.model = model
        self.mode = mode
        self.class_names = class_names
        
        if mode == 'image':
            self.explainer = lime.lime_image.LimeImageExplainer()
        elif mode == 'tabular':
            if training_data is None:
                raise ValueError("Tabular mode requires training_data")
            if feature_names is None:
                feature_names = [f'f_{i}' for i in range(training_data.shape[1])]
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data, feature_names=feature_names,
                class_names=class_names, mode='classification'
            )
    
    def _predict_images(self, images: np.ndarray) -> np.ndarray:
        """Prediction function for image LIME."""
        self.model.eval()
        images = images.astype(np.float32) / 255.0
        images = np.transpose(images, (0, 3, 1, 2))
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images = (images - mean) / std
        
        with torch.no_grad():
            x = torch.FloatTensor(images)
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
        return probs.numpy()
    
    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        """Prediction function for tabular LIME."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features)
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
        return probs.numpy()
    
    def explain_image(self, image: np.ndarray, num_samples: int = 1000,
                      num_features: int = 10):
        """Explain an image classification."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        return self.explainer.explain_instance(
            image, self._predict_images, top_labels=3,
            num_samples=num_samples, num_features=num_features
        )
    
    def explain_features(self, features: np.ndarray, num_samples: int = 1000,
                         num_features: int = 10):
        """Explain a feature-based prediction."""
        return self.explainer.explain_instance(
            features, self._predict_features,
            num_samples=num_samples, num_features=num_features
        )


def explain_with_lime(model: nn.Module, image: np.ndarray,
                      class_names: Optional[List[str]] = None,
                      num_samples: int = 1000) -> Tuple[np.ndarray, Dict]:
    """Convenience function for LIME image explanation."""
    explainer = LIMEExplainer(model, mode='image', class_names=class_names)
    explanation = explainer.explain_image(image, num_samples=num_samples)
    pred_class = explanation.top_labels[0]
    viz, mask = explanation.get_image_and_mask(
        pred_class, positive_only=True, num_features=5, hide_rest=False
    )
    return viz, {'predicted_class': pred_class, 'mask': mask}
