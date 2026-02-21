"""
Classical SOTA Training Pipeline — ResNet50 Transfer Learning Baseline.

This is a state-of-the-art classical baseline designed to be genuinely
competitive with the Hybrid Quantum-Classical model, ensuring a fair
scientific comparison.

Architecture:
  ResNet50 (pretrained, ImageNet) → GlobalAvgPool [built in]
  → BatchNorm(2048) → Dropout(0.4)
  → Linear(2048 → 512) → ReLU
  → BatchNorm(512) → Dropout(0.3)
  → Linear(512 → 256) → ReLU
  → Dropout(0.2)
  → Linear(256 → num_classes)

Training Strategy (Two-Stage):
  Stage 1 — frozen backbone (10 epochs, LR 1e-3):  train classifier head only.
  Stage 2 — partial unfreeze (last ResNet block, 10 epochs, LR 5e-5):  fine-tune.

Techniques:
  - Cosine annealing LR scheduler
  - Label smoothing (0.1)
  - Heavy data augmentation (flips, rotation, color jitter)
  - Early stopping
  - Best checkpoint saving

Usage:
  python training/train_classical_sota.py \\
      --data_dir ./data/brain_mri \\
      --output_dir ./outputs/classical_sota \\
      --num_classes 4 \\
      --epochs_stage1 10 \\
      --epochs_stage2 15 \\
      --batch_size 32 \\
      --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report
)
from tqdm import tqdm


# ─────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────

class ClassicalSOTAModel(nn.Module):
    """
    Strong classical baseline using ResNet50 + deep classification head.

    Designed to give the hybrid quantum model a genuinely hard reference point.
    The head is deliberately deeper than the hybrid model's classical branch.
    """

    def __init__(self, num_classes: int = 4, dropout_head: float = 0.4):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────────────
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the original FC head — we keep everything up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → [B, 2048, 1, 1]

        # Freeze backbone by default (Stage 1)
        self._freeze_backbone()

        # ── Classification Head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),                          # [B, 2048]
            nn.BatchNorm1d(2048),
            nn.Dropout(p=dropout_head),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    # ── Freeze / Unfreeze helpers ─────────────────────────────────────────────

    def _freeze_backbone(self):
        """Freeze all backbone weights (Stage 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_block(self):
        """
        Unfreeze the last ResNet layer block (layer4) for Stage 2 fine-tuning.
        ResNet50 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        In nn.Sequential that becomes indices 0-8; layer4 is index 7.
        """
        for param in self.backbone[7].parameters():   # layer4
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze ResNet50 layer4. Trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Unfreeze entire backbone (aggressive fine-tune)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # [B, 2048, 1, 1]
        logits = self.head(features)  # [B, num_classes]
        return logits

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────

def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build strong train augmentation and clean val/test transforms."""
    train_tf = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test DataLoaders from an ImageFolder dataset."""
    train_tf, val_tf = build_transforms(image_size)

    # Load full dataset with train transforms, then split
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    class_names = full_dataset.classes
    n = len(full_dataset)

    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Override transform for val/test (no augmentation)
    val_ds.dataset  = datasets.ImageFolder(data_dir, transform=val_tf)
    test_ds.dataset = datasets.ImageFolder(data_dir, transform=val_tf)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    print(f"\nDataset: {data_dir}")
    print(f"  Classes  : {class_names}")
    print(f"  Train    : {len(train_ds):,} samples")
    print(f"  Val      : {len(val_ds):,} samples")
    print(f"  Test     : {len(test_ds):,} samples")

    return train_loader, val_loader, test_loader, class_names


# ─────────────────────────────────────────────────────────
# TRAIN / VALIDATE HELPERS
# ─────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    grad_clip: float = 1.0,
    phase: str = "train",
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one epoch (train or eval).

    Returns:
        avg_loss, accuracy, all_labels, all_preds, all_probs
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc=phase.capitalize(), leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ─────────────────────────────────────────────────────────
# TWO-STAGE TRAINER
# ─────────────────────────────────────────────────────────

def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: str,
    output_dir: Path,
    stage_name: str = "stage",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    early_stopping_patience: int = 8,
    grad_clip: float = 1.0,
) -> Dict[str, List]:
    """Train for one stage and return history. Saves best checkpoint per stage."""

    print(f"\n{'='*60}")
    print(f"  {stage_name.upper()}")
    print(f"  Epochs: {num_epochs}  |  LR: {lr}  |  Label smoothing: {label_smoothing}")
    params = model.count_parameters()
    print(f"  Trainable params: {params['trainable']:,} / {params['total']:,}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_ckpt_path = output_dir / f"best_{stage_name}.pt"

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc, _, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=grad_clip, phase="train"
        )
        val_loss, val_acc, _, _, _ = run_epoch(
            model, val_loader, criterion, None, device, phase="val"
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_ckpt_path)
            marker = "  ✅ best"
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        print(
            f"[{stage_name}] Epoch {epoch:>3}/{num_epochs} | "
            f"Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"LR {current_lr:.2e} | {elapsed:.1f}s{marker}"
        )

        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping triggered after {epoch} epochs.")
            break

    print(f"\n  {stage_name} complete. Best val acc: {best_val_acc:.4f}")
    return history


def train_classical_sota(
    data_dir: str,
    output_dir: str,
    num_classes: int = 4,
    epochs_stage1: int = 10,
    epochs_stage2: int = 15,
    batch_size: int = 32,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    image_size: int = 224,
) -> None:
    """
    Full two-stage SOTA training pipeline.

    Stage 1: Frozen backbone, train head only (high LR).
    Stage 2: Unfreeze layer4, fine-tune end-to-end (low LR).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _resolve_device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers,
        image_size=image_size, seed=seed
    )
    num_classes = len(class_names) if num_classes is None else num_classes

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ClassicalSOTAModel(num_classes=num_classes).to(device)
    print(f"\nModel: ClassicalSOTAModel (ResNet50 backbone)")
    params = model.count_parameters()
    print(f"  Total params   : {params['total']:,}")
    print(f"  Trainable      : {params['trainable']:,}")
    print(f"  Frozen backbone: {params['frozen']:,}")

    # ── Stage 1: Train head with frozen backbone ───────────────────────────────
    history_s1 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs_stage1,
        lr=1e-3,
        device=device,
        output_dir=out,
        stage_name="stage1_frozen",
        early_stopping_patience=6,
    )

    # ── Load best stage1, unfreeze layer4 for Stage 2 ─────────────────────────
    ckpt_s1 = torch.load(out / "best_stage1_frozen.pt", map_location=device)
    model.load_state_dict(ckpt_s1['model_state_dict'])
    model.unfreeze_last_block()

    # ── Stage 2: Fine-tune last block ─────────────────────────────────────────
    history_s2 = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs_stage2,
        lr=5e-5,
        device=device,
        output_dir=out,
        stage_name="stage2_finetune",
        early_stopping_patience=8,
    )

    # ── Load best stage2 & save as final best_model.pt ───────────────────────
    ckpt_s2 = torch.load(out / "best_stage2_finetune.pt", map_location=device)
    model.load_state_dict(ckpt_s2['model_state_dict'])
    torch.save({'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'backbone': 'resnet50'},
               out / "best_model.pt")
    print(f"\n  Saved final model → {out / 'best_model.pt'}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    criterion_eval = nn.CrossEntropyLoss()
    _, _, y_true, y_pred, y_prob = run_epoch(
        model, test_loader, criterion_eval, None, device, phase="test"
    )
    metrics = _compute_full_metrics(y_true, y_pred, y_prob, class_names)

    print("\n" + "="*60)
    print("  FINAL TEST RESULTS — Classical SOTA (ResNet50)")
    print("="*60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Precision : {metrics['precision_macro']:.4f}")
    print(f"  Recall    : {metrics['recall_macro']:.4f}")
    print("="*60)
    print("\n" + metrics['classification_report'])

    # ── Merge histories & save ────────────────────────────────────────────────
    combined_history = {
        'stage1': history_s1,
        'stage2': history_s2,
        'test_metrics': metrics,
        'class_names': class_names,
    }
    history_path = out / "history.json"
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2, default=str)
    print(f"\n  Saved training history → {history_path}")


# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU.")
        return "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, falling back to CPU.")
        return "cpu"
    return device


def _compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    # ROC-AUC (OvR, macro)
    try:
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    return {
        'accuracy': float(acc),
        'f1_macro': float(f1_mac),
        'f1_per_class': f1_per,
        'precision_macro': float(prec),
        'recall_macro': float(rec),
        'roc_auc': float(auc),
        'classification_report': report,
    }


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Classical SOTA ResNet50 baseline")
    p.add_argument("--data_dir",       type=str,   required=True,
                   help="Path to ImageFolder dataset root")
    p.add_argument("--output_dir",     type=str,   default="./outputs/classical_sota")
    p.add_argument("--num_classes",    type=int,   default=None,
                   help="Number of classes (auto-detected if omitted)")
    p.add_argument("--epochs_stage1",  type=int,   default=10,
                   help="Epochs for Stage 1 (frozen backbone)")
    p.add_argument("--epochs_stage2",  type=int,   default=15,
                   help="Epochs for Stage 2 (fine-tune layer4)")
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--device",         type=str,   default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--image_size",     type=int,   default=224)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classical_sota(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        image_size=args.image_size,
    )
