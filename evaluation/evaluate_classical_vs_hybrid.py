"""
Head-to-Head Evaluation: Classical SOTA vs Hybrid Quantum-Classical Model.

Loads both checkpoints, evaluates on the **same** test set,
and prints a side-by-side metrics comparison.

Usage:
  python evaluation/evaluate_classical_vs_hybrid.py \\
      --data_dir ./data/brain_mri \\
      --hybrid_checkpoint ./outputs/high_perf_resnet/checkpoints/best_model.pt \\
      --classical_checkpoint ./outputs/classical_sota/best_model.pt \\
      --batch_size 32 \\
      --device cuda
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
    confusion_matrix
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────

def build_test_loader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, List[str]]:
    """
    Build a deterministic test DataLoader using the same split logic
    as train_classical_sota.py so the test set is identical across runs.
    """
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    full_ds = datasets.ImageFolder(data_dir, transform=val_tf)
    class_names = full_ds.classes
    n = len(full_ds)

    n_test  = int(n * test_ratio)
    n_val   = int(n * 0.15)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    _, _, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

    pin_memory = torch.cuda.is_available()
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    return loader, class_names


# ─────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────

def load_classical_sota_model(checkpoint_path: str, device: str) -> nn.Module:
    """Load the ClassicalSOTAModel from training/train_classical_sota.py."""
    from training.train_classical_sota import ClassicalSOTAModel

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get('num_classes', 4)
    model = ClassicalSOTAModel(num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✅ Loaded Classical SOTA   : {checkpoint_path}")
    return model


def load_hybrid_model(checkpoint_path: str, device: str) -> nn.Module:
    """
    Load the HybridQuantumClassifier using the saved config inside the checkpoint.
    Falls back to default Config if no config is saved.
    """
    from configs.config import Config, ModelConfig, QuantumConfig, TrainingConfig
    from models.hybrid_model import HybridQuantumClassifier

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to reconstruct config from checkpoint
    if 'config' in ckpt:
        raw_cfg = ckpt['config']
        # Build minimal Config by manually patching key fields
        config = Config()
        if 'model' in raw_cfg:
            for k, v in raw_cfg['model'].items():
                if hasattr(config.model, k):
                    try:
                        setattr(config.model, k, v)
                    except Exception:
                        pass
        if 'quantum' in raw_cfg:
            for k, v in raw_cfg['quantum'].items():
                if hasattr(config.quantum, k):
                    try:
                        setattr(config.quantum, k, v)
                    except Exception:
                        pass
        if 'training' in raw_cfg:
            for k, v in raw_cfg['training'].items():
                if hasattr(config.training, k):
                    try:
                        setattr(config.training, k, v)
                    except Exception:
                        pass
    else:
        config = Config()

    model = HybridQuantumClassifier(config, use_quantum=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  ✅ Loaded Hybrid QML model : {checkpoint_path}")
    return model


# ─────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    model_name: str = "Model",
) -> Dict:
    """Run evaluation and collect all predictions, probabilities, and labels."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc=f"Evaluating {model_name}", leave=True):
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        preds  = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    return {'y_true': y_true, 'y_pred': y_pred, 'y_prob': y_prob}


def compute_metrics(y_true, y_pred, y_prob, class_names) -> Dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    rep  = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    try:
        n_cls = y_prob.shape[1]
        if n_cls == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    return {
        'accuracy': float(acc),
        'f1_macro': float(f1),
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'roc_auc': float(auc),
        'classification_report': rep,
        'confusion_matrix': cm.tolist(),
    }


# ─────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────

def print_comparison(
    hybrid_metrics: Dict,
    classical_metrics: Dict,
    hybrid_name: str = "Hybrid QML  (ResNet18+Quantum)",
    classical_name: str = "Classical SOTA (ResNet50)",
) -> None:
    """Print a formatted side-by-side comparison table."""
    W = 60

    def row(label, h_val, c_val, fmt=".4f"):
        h_str = format(h_val, fmt) if isinstance(h_val, (int, float)) and not np.isnan(h_val) else str(h_val)
        c_str = format(c_val, fmt) if isinstance(c_val, (int, float)) and not np.isnan(c_val) else str(c_val)
        winner = "◀ Hybrid" if h_val > c_val else ("▶ Classical" if c_val > h_val else "  Tied")
        print(f"  {label:<22} {h_str:>10}    {c_str:>10}    {winner}")

    print("\n" + "="*W)
    print("  MODEL COMPARISON — Hybrid QML  vs  Classical SOTA")
    print("="*W)
    print(f"  {'Metric':<22} {'Hybrid':>10}    {'Classical':>10}    Winner")
    print("  " + "-"*(W-2))
    row("Accuracy",      hybrid_metrics['accuracy'],         classical_metrics['accuracy'])
    row("F1 (macro)",    hybrid_metrics['f1_macro'],         classical_metrics['f1_macro'])
    row("ROC-AUC",       hybrid_metrics['roc_auc'],          classical_metrics['roc_auc'])
    row("Precision",     hybrid_metrics['precision_macro'],  classical_metrics['precision_macro'])
    row("Recall",        hybrid_metrics['recall_macro'],     classical_metrics['recall_macro'])
    print("="*W)

    print(f"\n──── {hybrid_name} ─────────────────────")
    print(hybrid_metrics['classification_report'])

    print(f"\n──── {classical_name} ──────────────────")
    print(classical_metrics['classification_report'])


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Side-by-side evaluation: Hybrid vs Classical SOTA")
    p.add_argument("--data_dir",              type=str, required=True)
    p.add_argument("--hybrid_checkpoint",     type=str, required=True,
                   help="Path to hybrid best_model.pt")
    p.add_argument("--classical_checkpoint",  type=str, required=True,
                   help="Path to classical_sota best_model.pt")
    p.add_argument("--batch_size",            type=int, default=32)
    p.add_argument("--num_workers",           type=int, default=4)
    p.add_argument("--device",                type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--seed",                  type=int, default=42)
    p.add_argument("--output_json",           type=str, default=None,
                   help="Optional path to save comparison results as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("⚠️  CUDA unavailable, using CPU.")
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("⚠️  MPS unavailable, using CPU.")

    print(f"\nDevice: {device}")
    print(f"Data  : {args.data_dir}")
    print(f"Hybrid checkpoint     : {args.hybrid_checkpoint}")
    print(f"Classical checkpoint  : {args.classical_checkpoint}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    test_loader, class_names = build_test_loader(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed
    )
    print(f"Test samples: {len(test_loader.dataset)}  |  Classes: {class_names}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    hybrid_model    = load_hybrid_model(args.hybrid_checkpoint, device)
    classical_model = load_classical_sota_model(args.classical_checkpoint, device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    hybrid_raw    = evaluate_model(hybrid_model,    test_loader, device, "Hybrid QML")
    classical_raw = evaluate_model(classical_model, test_loader, device, "Classical SOTA")

    hybrid_metrics    = compute_metrics(*hybrid_raw.values(), class_names)
    classical_metrics = compute_metrics(*classical_raw.values(), class_names)

    # ── Print comparison ──────────────────────────────────────────────────────
    print_comparison(hybrid_metrics, classical_metrics)

    # ── Optionally save JSON ──────────────────────────────────────────────────
    if args.output_json:
        results = {
            'hybrid': {k: v for k, v in hybrid_metrics.items() if k != 'classification_report'},
            'classical': {k: v for k, v in classical_metrics.items() if k != 'classification_report'},
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved → {args.output_json}")


if __name__ == "__main__":
    main()
