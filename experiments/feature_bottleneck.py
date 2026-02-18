#!/usr/bin/env python
"""
Feature Dimension Bottleneck Experiment.

Tests whether the ≤10 feature compression layer (designed for quantum encoding)
creates an information bottleneck that limits classification performance.

Trains classical-only CNN classifiers with different compression dimensions:
  - 10 features (current quantum-compatible setting)
  - 32 features
  - 64 features

All models use the SAME CNN backbone and training pipeline — only the
compression output dimension and classifier input dimension change.

Usage:
    python experiments/feature_bottleneck.py --data_dir ./data/brain_mri --epochs 50
    python experiments/feature_bottleneck.py --data_dir ./data/brain_mri --epochs 50 --device mps
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config, ModelConfig, DataConfig, TrainingConfig
from data.dataset import get_dataloaders
from models.cnn_feature_extractor import ConvBlock
from training.trainer import Trainer
from training.metrics import compute_metrics


# ============================================================
# Custom CNN that allows arbitrary compression dimensions
# ============================================================

class FlexibleCNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor with configurable compression dimension.
    
    Same architecture as CNNFeatureExtractor but WITHOUT the ≤10 
    quantum constraint, allowing experiments with any feature dimension.
    """
    
    def __init__(self, config: ModelConfig, input_channels: int = 3):
        super().__init__()
        
        self.output_dim = config.num_features  # No ≤10 restriction
        
        conv_channels = config.conv_channels
        kernel_size = config.kernel_size
        pool_size = config.pool_size
        use_batch_norm = config.use_batch_norm
        dropout_rate = config.dropout_rate
        
        # Same conv blocks as original CNN
        self.conv1 = ConvBlock(input_channels, conv_channels[0],
                               kernel_size, pool_size, use_batch_norm, dropout_rate)
        self.conv2 = ConvBlock(conv_channels[0], conv_channels[1],
                               kernel_size, pool_size, use_batch_norm, dropout_rate)
        self.conv3 = ConvBlock(conv_channels[1], conv_channels[2],
                               kernel_size, pool_size, use_batch_norm, dropout_rate)
        self.conv4 = ConvBlock(conv_channels[2], conv_channels[3],
                               kernel_size, pool_size, use_batch_norm, dropout_rate)
        
        self.features = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Flexible compression — adapts intermediate dim based on output
        intermediate_dim = max(64, self.output_dim * 2)
        self.compression = nn.Sequential(
            nn.Linear(conv_channels[-1], intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, self.output_dim),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.gap(features)
        flattened = pooled.view(pooled.size(0), -1)
        compressed = self.compression(flattened)
        return compressed


class ClassicalClassifier(nn.Module):
    """
    Classical-only classifier with flexible feature dimension.

    Architecture: FlexibleCNN → Dense Head → Logits
    No quantum layer — purely classical for bottleneck analysis.
    """
    
    def __init__(self, config: ModelConfig, num_classes: int, input_channels: int = 3):
        super().__init__()
        
        self.num_features = config.num_features
        self.num_classes = num_classes
        
        # CNN with flexible compression
        self.cnn = FlexibleCNNFeatureExtractor(config, input_channels)
        
        # Classifier head (same structure as hybrid model)
        self.classifier = nn.Sequential(
            nn.Linear(config.num_features, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(32, num_classes),
        )
        
        print(f"  ClassicalClassifier: features={config.num_features}, classes={num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        logits = self.classifier(features)
        return logits


# ============================================================
# Experiment runner
# ============================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_config(args, num_features: int) -> Config:
    """
    Build a Config that bypasses quantum constraints.
    
    We override __post_init__ validation by setting n_qubits = num_features
    even though we won't use the quantum layer.
    """
    config = Config.__new__(Config)
    
    config.data = DataConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    config.model = ModelConfig(
        num_features=min(num_features, 10),  # Stays ≤10 for Config validation
        num_classes=3,  # Will be overridden from dataset
    )
    
    config.quantum = Config().quantum  # Default quantum config
    config.quantum.n_qubits = config.model.num_features
    config.quantum.n_outputs = config.model.num_features
    
    config.training = TrainingConfig(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        checkpoint_dir=f'./outputs/bottleneck_experiment/checkpoints_{num_features}',
        log_dir=f'./outputs/bottleneck_experiment/logs_{num_features}',
    )
    
    config.xgboost = Config().xgboost
    config.xai = Config().xai
    config.project_name = 'bottleneck_experiment'
    config.experiment_name = f'features_{num_features}'
    config.output_dir = f'./outputs/bottleneck_experiment'
    config.seed = args.seed
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.training.log_dir).mkdir(parents=True, exist_ok=True)
    
    return config


def train_and_evaluate(
    num_features: int,
    args,
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    device: str,
) -> dict:
    """Train a classical model with given feature dimension and return results."""
    
    print(f"\n{'='*60}")
    print(f"  TRAINING: {num_features} COMPRESSED FEATURES (classical-only)")
    print(f"{'='*60}")
    
    # Build config (only used for training hyperparameters)
    config = build_config(args, num_features)
    config.model.num_classes = num_classes
    
    # Create model with ACTUAL num_features (bypassing the ≤10 check)
    model_config = ModelConfig(
        num_features=num_features,  # The real dimension we want to test
        num_classes=num_classes,
        dropout_rate=config.model.dropout_rate,
        conv_channels=config.model.conv_channels,
        kernel_size=config.model.kernel_size,
        pool_size=config.model.pool_size,
        use_batch_norm=config.model.use_batch_norm,
    )
    
    model = ClassicalClassifier(model_config, num_classes)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Train using the project's Trainer
    trainer = Trainer(model, config, device)
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, num_epochs=args.epochs)
    train_time = time.time() - start_time
    
    # Evaluate on test set
    model.eval()
    model.to(device)
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_metrics = compute_metrics(all_labels, all_preds, np.array(all_probs))
    
    result = {
        'num_features': num_features,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_time_seconds': round(train_time, 1),
        'best_val_accuracy': trainer.best_val_accuracy,
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': {k: round(v, 4) for k, v in test_metrics.items()},
        'history': {
            'train_loss': [float(v) for v in history['train_loss']],
            'val_loss': [float(v) for v in history['val_loss']],
            'train_accuracy': [float(v) for v in history['train_accuracy']],
            'val_accuracy': [float(v) for v in history['val_accuracy']],
        }
    }
    
    print(f"\n  Results for {num_features} features:")
    print(f"    Best val accuracy:  {trainer.best_val_accuracy:.4f}")
    print(f"    Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"    Test F1:            {test_metrics.get('f1', 0):.4f}")
    print(f"    Training time:      {train_time:.1f}s")
    
    return result


def plot_results(results: list, save_dir: str):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    dims = [r['num_features'] for r in results]
    val_accs = [r['best_val_accuracy'] for r in results]
    test_accs = [r['test_metrics']['accuracy'] for r in results]
    params = [r['trainable_params'] for r in results]
    
    # --- Plot 1: Feature Dimension vs Accuracy ---
    ax1 = axes[0]
    x = np.arange(len(dims))
    width = 0.35
    bars1 = ax1.bar(x - width/2, [v * 100 for v in val_accs], width,
                     label='Best Val Accuracy', color='#4A90D9', edgecolor='white')
    bars2 = ax1.bar(x + width/2, [v * 100 for v in test_accs], width,
                     label='Test Accuracy', color='#E8734A', edgecolor='white')
    ax1.set_xlabel('Feature Compression Dimension', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Feature Dimension vs Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    
    # --- Plot 2: Training curves (val accuracy) ---
    ax2 = axes[1]
    colors = ['#4A90D9', '#50C878', '#E8734A']
    for i, r in enumerate(results):
        epochs = range(1, len(r['history']['val_accuracy']) + 1)
        ax2.plot(epochs, [v * 100 for v in r['history']['val_accuracy']],
                 label=f"{r['num_features']} features", color=colors[i % len(colors)], linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy Over Training', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    # --- Plot 3: Parameters vs Accuracy tradeoff ---
    ax3 = axes[2]
    for i, r in enumerate(results):
        ax3.scatter(r['trainable_params'], r['test_metrics']['accuracy'] * 100,
                    s=200, color=colors[i % len(colors)], zorder=5, edgecolors='black')
        ax3.annotate(f"{r['num_features']}d",
                     (r['trainable_params'], r['test_metrics']['accuracy'] * 100),
                     textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight='bold')
    ax3.set_xlabel('Trainable Parameters', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Parameter Efficiency', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = f'{save_dir}/feature_bottleneck_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Feature Dimension Bottleneck Experiment'
    )
    parser.add_argument('--data_dir', type=str, default='./data/brain_mri')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps). Auto-detected if not set.')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 32, 64],
                        help='Feature dimensions to test (default: 10 32 64)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("="*60)
    print("  FEATURE DIMENSION BOTTLENECK EXPERIMENT")
    print("="*60)
    print(f"  Dimensions to test: {args.dims}")
    print(f"  Epochs per model:   {args.epochs}")
    print(f"  Device:             {device}")
    print(f"  Seed:               {args.seed}")
    print("="*60)
    
    # Load data once (shared across all experiments)
    base_config = build_config(args, 10)
    train_loader, val_loader, test_loader, dataset = get_dataloaders(base_config)
    num_classes = len(dataset.classes)
    base_config.model.num_classes = num_classes
    print(f"\nClasses ({num_classes}): {dataset.classes}")
    
    save_dir = './outputs/bottleneck_experiment'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = []
    for dim in args.dims:
        set_seed(args.seed)  # Reset seed for fair comparison
        result = train_and_evaluate(
            num_features=dim,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            device=device,
        )
        all_results.append(result)
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "="*70)
    print("  EXPERIMENT SUMMARY: Feature Dimension Bottleneck Analysis")
    print("="*70)
    print(f"{'Dim':<8} {'Params':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Time':<10}")
    print("-"*70)
    for r in all_results:
        print(
            f"{r['num_features']:<8} "
            f"{r['trainable_params']:<12,} "
            f"{r['best_val_accuracy']:<12.4f} "
            f"{r['test_metrics']['accuracy']:<12.4f} "
            f"{r['test_metrics'].get('f1', 0):<12.4f} "
            f"{r['train_time_seconds']:<10.1f}s"
        )
    print("="*70)
    
    # Bottleneck analysis
    best = max(all_results, key=lambda r: r['test_metrics']['accuracy'])
    baseline = next(r for r in all_results if r['num_features'] == 10)
    improvement = best['test_metrics']['accuracy'] - baseline['test_metrics']['accuracy']
    
    if improvement > 0.02:
        print(f"\n⚠️  BOTTLENECK DETECTED: {best['num_features']}d features outperform "
              f"10d by {improvement*100:.1f}% test accuracy.")
        print(f"   The ≤10 compression for quantum encoding IS limiting performance.")
    elif improvement > 0:
        print(f"\n📊 MARGINAL DIFFERENCE: {best['num_features']}d features improve over "
              f"10d by only {improvement*100:.1f}% test accuracy.")
        print(f"   The ≤10 compression causes minimal information loss.")
    else:
        print(f"\n✅ NO BOTTLENECK: 10d features perform as well as or better than "
              f"larger dimensions.")
        print(f"   The ≤10 compression is NOT limiting performance.")
    
    # Save results
    results_path = f'{save_dir}/bottleneck_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Plot
    plot_results(all_results, save_dir)
    
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()
