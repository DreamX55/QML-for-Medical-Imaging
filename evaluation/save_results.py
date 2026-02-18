"""
Results Saving Utilities.

Provides functions to save evaluation results in various formats
for reproducibility and publication.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


def save_evaluation_results(
    results: Dict,
    save_dir: str,
    experiment_name: str = 'experiment',
    formats: List[str] = ['json', 'csv']
) -> List[str]:
    """
    Save evaluation results in multiple formats.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save
        experiment_name: Name for files
        formats: Output formats ('json', 'csv', 'latex')
        
    Returns:
        List of saved file paths
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved_files = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Clean results for serialization
    clean_results = _clean_for_json(results)
    
    if 'json' in formats:
        json_path = f'{save_dir}/{experiment_name}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        saved_files.append(json_path)
        print(f"Saved: {json_path}")
    
    if 'csv' in formats and 'metrics' in results:
        csv_path = f'{save_dir}/{experiment_name}_{timestamp}.csv'
        metrics = results['metrics']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for k, v in metrics.items():
                writer.writerow([k, v])
        saved_files.append(csv_path)
        print(f"Saved: {csv_path}")
    
    if 'latex' in formats and 'metrics' in results:
        latex_path = f'{save_dir}/{experiment_name}_{timestamp}.tex'
        _save_latex_table(results['metrics'], latex_path)
        saved_files.append(latex_path)
        print(f"Saved: {latex_path}")
    
    return saved_files


def _clean_for_json(obj: Any) -> Any:
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    return obj


def _save_latex_table(metrics: Dict, path: str) -> None:
    """Save metrics as LaTeX table."""
    with open(path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\hline\n")
        f.write("Metric & Value \\\\\n")
        f.write("\\hline\n")
        for k, v in metrics.items():
            f.write(f"{k.replace('_', ' ').title()} & {v:.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Model Evaluation Results}\n")
        f.write("\\end{table}\n")


def create_results_report(
    results: Dict,
    config: Dict,
    save_path: str
) -> None:
    """
    Create comprehensive results report in Markdown.
    
    Args:
        results: Evaluation results
        config: Configuration used
        save_path: Path to save report
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("# Experiment Results Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        for key, value in config.items():
            if isinstance(value, dict):
                f.write(f"{key}:\n")
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("```\n\n")
        
        f.write("## Results\n\n")
        if 'metrics' in results:
            f.write("### Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for k, v in results['metrics'].items():
                f.write(f"| {k} | {v:.4f} |\n")
            f.write("\n")
        
        if 'per_class_metrics' in results:
            f.write("### Per-Class Metrics\n\n")
            for cls, metrics in results['per_class_metrics'].items():
                f.write(f"**{cls}**: P={metrics['precision']:.3f}, ")
                f.write(f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}\n\n")
    
    print(f"Report saved: {save_path}")


def save_training_curves(
    history: Dict,
    save_path: str
) -> None:
    """Save training curves plot."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    # Accuracy
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
