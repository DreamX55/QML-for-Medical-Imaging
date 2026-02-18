"""
Evaluation module for model analysis and comparison.

Provides evaluation scripts, ablation studies, and result saving.
"""

from .evaluate import evaluate_model, evaluate_on_test_set
from .ablation import run_ablation_study, compare_models
from .feature_importance import analyze_feature_importance
from .save_results import save_evaluation_results, create_results_report

__all__ = [
    "evaluate_model",
    "evaluate_on_test_set",
    "run_ablation_study",
    "compare_models",
    "analyze_feature_importance",
    "save_evaluation_results",
    "create_results_report",
]
