"""
Explainable AI (XAI) Module for Medical Image Classification.

Provides interpretability tools including:
- Grad-CAM for CNN visualization
- SHAP for feature importance
- LIME for local explanations
- Unified visualization utilities
"""

from .gradcam import GradCAM, compute_gradcam
from .shap_explainer import SHAPExplainer, explain_with_shap
from .lime_explainer import LIMEExplainer, explain_with_lime
from .visualizations import (
    plot_gradcam_overlay,
    plot_shap_summary,
    plot_lime_explanation,
    create_explanation_report,
)

__all__ = [
    "GradCAM",
    "compute_gradcam",
    "SHAPExplainer",
    "explain_with_shap",
    "LIMEExplainer",
    "explain_with_lime",
    "plot_gradcam_overlay",
    "plot_shap_summary",
    "plot_lime_explanation",
    "create_explanation_report",
]
