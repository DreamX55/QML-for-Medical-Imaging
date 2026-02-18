#!/usr/bin/env python
"""
Main Entry Point for Hybrid Quantum-Classical Medical Image Classification.

This script provides a unified interface for:
- Training the hybrid model
- Evaluating models on test data
- Running XAI explanations
- Performing ablation studies

Usage:
    python main.py --mode train --data_dir ./data/brain_mri
    python main.py --mode evaluate --checkpoint ./outputs/checkpoints/best_model.pt
    python main.py --mode explain --checkpoint ./outputs/checkpoints/best_model.pt --image ./sample.jpg
    python main.py --mode ablation --data_dir ./data/brain_mri
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import Config
from data.dataset import get_dataloaders
from models.hybrid_model import create_model, count_parameters
from training.trainer import Trainer
from evaluation.evaluate import evaluate_on_test_set
from evaluation.ablation import run_ablation_study
from evaluation.feature_importance import analyze_feature_importance


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hybrid Quantum-Classical Medical Image Classification'
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'finetune', 'evaluate', 'explain', 'ablation', 'xgboost'],
        help='Operation mode'
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data/brain_mri',
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['hybrid', 'classical', 'ensemble'],
                        help='Type of model to use')
    parser.add_argument('--num_features', type=int, default=10,
                        help='Number of CNN output features (max 10)')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of output classes')
    
    # Quantum arguments
    parser.add_argument('--n_qubits', type=int, default=10,
                        help='Number of qubits')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of quantum layers')
    parser.add_argument('--entanglement', type=str, default='linear',
                        choices=['linear', 'ring', 'full'],
                        help='Entanglement pattern')
    parser.add_argument('--quantum_device', type=str, default='default.qubit',
                        choices=['default.qubit', 'lightning.qubit'],
                        help='Quantum simulator backend')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--quantum_lr', type=float, default=1e-2,
                        help='Quantum parameters learning rate')
    
    # Checkpoints
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation/explanation')
    parser.add_argument('--classical_checkpoint', type=str, default=None,
                        help='Path to classical checkpoint for finetune mode')
    parser.add_argument('--freeze_cnn', action='store_true',
                        help='Freeze CNN backbone during fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    
    # XAI arguments
    parser.add_argument('--image', type=str, default=None,
                        help='Image path for explanation')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu/mps)')
    
    return parser.parse_args()


def build_config(args) -> Config:
    """Build configuration from arguments."""
    config = Config(
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Data config
    config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    # Model config
    config.model.num_features = args.num_features
    config.model.num_classes = args.num_classes
    
    # Quantum config
    config.quantum.n_qubits = args.n_qubits
    config.quantum.n_layers = args.n_layers
    config.quantum.entanglement = args.entanglement
    config.quantum.device = args.quantum_device
    
    # Training config
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.quantum_learning_rate = args.quantum_lr
    
    return config


def train_model(args, config: Config):
    """Train the hybrid model."""
    print("\n" + "="*60)
    print("TRAINING HYBRID QUANTUM-CLASSICAL MODEL")
    print("="*60)
    
    # Set seed
    set_seed(config.seed)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    config.model.num_classes = len(dataset.classes)
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(config, model_type=args.model_type)
    params = count_parameters(model)
    print(f"Model parameters: {params}")
    
    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_on_test_set(
        model, test_loader, device, class_names,
        save_path=f'{config.output_dir}/test_results.json'
    )
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(
        model, train_loader, test_loader, config, device,
        save_dir=f'{config.output_dir}/feature_analysis'
    )
    
    print("\nTraining complete!")
    return model, history, results


def evaluate_model(args, config: Config):
    """Evaluate a trained model from a saved checkpoint."""
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    if args.checkpoint is None:
        raise ValueError("--checkpoint required for evaluation")
    
    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), (
        f"Checkpoint not found: {checkpoint_path}\n"
        f"Run training first or provide a valid --checkpoint path."
    )
    
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    # Load data
    _, _, test_loader, dataset = get_dataloaders(config)
    config.model.num_classes = len(dataset.classes)
    
    # ── Step 1: Create model with matching architecture ──
    model = create_model(config, model_type=args.model_type)
    
    # Snapshot parameter norms BEFORE loading checkpoint.
    # After loading, these must differ — otherwise the checkpoint
    # didn't actually overwrite the random initialization.
    init_norms = {
        name: p.data.norm().item()
        for name, p in model.named_parameters()
    }
    
    # ── Step 2: Load checkpoint with strict key matching ──
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint metadata if available
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'best_val_accuracy' in checkpoint:
        print(f"  Checkpoint val accuracy: {checkpoint['best_val_accuracy']:.4f}")
    if 'best_val_loss' in checkpoint:
        print(f"  Checkpoint val loss: {checkpoint['best_val_loss']:.4f}")
    
    # strict=True ensures every key in the state_dict matches the model.
    # If model architecture doesn't match the saved checkpoint, this will
    # raise a RuntimeError listing the mismatched keys instead of silently
    # loading partial weights.
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("  ✅ State dict loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"\n  🔴 ARCHITECTURE MISMATCH:")
        print(f"     Model and checkpoint have different parameter shapes.")
        print(f"     Ensure config matches the training configuration.")
        print(f"     Error: {e}")
        raise
    
    # ── Step 3: Verify weights actually changed from initialization ──
    loaded_norms = {
        name: p.data.norm().item()
        for name, p in model.named_parameters()
    }
    
    changed = sum(
        1 for name in init_norms
        if abs(init_norms[name] - loaded_norms[name]) > 1e-6
    )
    total = len(init_norms)
    print(f"  Parameters changed from init: {changed}/{total}")
    
    assert changed > 0, (
        "🔴 CHECKPOINT LOADING FAILED: No parameters changed from random "
        "initialization! The checkpoint may be corrupt or empty."
    )
    
    # ── Step 4: Set eval mode BEFORE any inference ──
    # This disables dropout and sets batch norm to use running stats.
    # Must be called AFTER load_state_dict, BEFORE any forward pass.
    model.eval()
    model.to(device)
    print("  model.eval() set ✅")
    
    # Guard: ensure eval mode is active
    assert not model.training, (
        "model.training is True after model.eval()! Something reinitialized the model."
    )
    
    # ── Step 5: Evaluate (no reinitialization after this point) ──
    results = evaluate_on_test_set(
        model, test_loader, device, dataset.classes,
        save_path=f'{config.output_dir}/evaluation_results.json'
    )
    
    return results


def explain_prediction(args, config: Config):
    """Generate explanations for a prediction."""
    print("\n" + "="*60)
    print("GENERATING EXPLANATIONS")
    print("="*60)
    
    if args.checkpoint is None:
        raise ValueError("--checkpoint required for explanation")
    if args.image is None:
        raise ValueError("--image required for explanation")
    
    from PIL import Image
    from data.transforms import get_val_transforms, inverse_normalize
    from xai.gradcam import GradCAM, overlay_cam_on_image
    from xai.visualizations import create_explanation_report
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(config, model_type=args.model_type)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = get_val_transforms(config.data)
    img = Image.open(args.image).convert('RGB')
    img_np = np.array(img)
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax().item()
        confidence = probs[0, pred_class].item()
    
    print(f"Prediction: Class {pred_class}, Confidence: {confidence:.2%}")
    
    # Grad-CAM
    gradcam = GradCAM(model)
    heatmap = gradcam(input_tensor, pred_class)
    
    # SHAP on features
    cnn_features = model.get_cnn_features(input_tensor).detach().cpu().numpy()
    shap_importance = {f'feature_{i}': float(abs(v)) for i, v in enumerate(cnn_features[0])}
    
    # Create report
    save_path = f'{config.output_dir}/explanations/report.png'
    create_explanation_report(
        img_np, heatmap, shap_importance,
        prediction=f'Class {pred_class}',
        confidence=confidence,
        save_path=save_path
    )
    
    print(f"\nExplanation saved to {save_path}")


def run_ablation(args, config: Config):
    """Run ablation study."""
    print("\n" + "="*60)
    print("RUNNING ABLATION STUDY")
    print("="*60)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    
    # Load data
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    config.model.num_classes = len(dataset.classes)
    
    # Run ablation
    results = run_ablation_study(
        config, train_loader, val_loader, test_loader,
        class_names=dataset.classes,
        device=device,
        save_dir=f'{config.output_dir}/ablation'
    )
    
    return results


def run_xgboost_comparison(args, config: Config):
    """Run XGBoost comparison on CNN features."""
    print("\n" + "="*60)
    print("RUNNING XGBOOST COMPARISON")
    print("="*60)
    
    from models.xgboost_classifier import XGBoostClassifier, extract_features_from_model
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config.seed)
    
    # Load data
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    config.model.num_classes = len(dataset.classes)
    n_classes = len(dataset.classes)
    
    # Create model for feature extraction
    model = create_model(config, model_type='classical')
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    model.to(device)
    model.eval()
    
    # Extract features
    print("\nExtracting CNN features...")
    train_cnn, train_labels = extract_features_from_model(
        model, train_loader, device, feature_type='cnn'
    )
    test_cnn, test_labels = extract_features_from_model(
        model, test_loader, device, feature_type='cnn'
    )
    
    # Train XGBoost
    print("\nTraining XGBoost classifier...")
    xgb_classifier = XGBoostClassifier(config.xgboost, n_classes)
    xgb_classifier.fit(train_cnn, train_labels)
    
    # Evaluate
    print("\nEvaluating XGBoost...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    predictions = xgb_classifier.predict(test_cnn)
    probabilities = xgb_classifier.predict_proba(test_cnn)
    
    results = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions, average='weighted'),
        'recall': recall_score(test_labels, predictions, average='weighted'),
        'f1': f1_score(test_labels, predictions, average='weighted'),
    }
    
    # Add AUC if possible
    try:
        results['auc'] = roc_auc_score(test_labels, probabilities, multi_class='ovr')
    except Exception:
        pass
    
    print("\n" + "="*50)
    print("XGBOOST RESULTS")
    print("="*50)
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    print("="*50)
    
    # Save results
    import json
    save_path = f'{config.output_dir}/xgboost_results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    return results


def finetune_model(args, config: Config):
    """Fine-tune a hybrid model from a pretrained classical checkpoint."""
    print("\n" + "="*60)
    print("STAGED FINE-TUNING: Classical CNN → Hybrid QML")
    print("="*60)
    
    # Validate args
    assert args.classical_checkpoint, (
        "--classical_checkpoint required for finetune mode"
    )
    assert os.path.exists(args.classical_checkpoint), (
        f"Classical checkpoint not found: {args.classical_checkpoint}"
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, dataset = get_dataloaders(config)
    config.model.num_classes = len(dataset.classes)
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    
    # Create HYBRID model
    print("\nCreating hybrid model...")
    model = create_model(config, model_type='hybrid')
    
    # Load classical CNN weights
    print(f"\nLoading classical checkpoint: {args.classical_checkpoint}")
    checkpoint = torch.load(args.classical_checkpoint, map_location='cpu')
    classical_state = checkpoint['model_state_dict']
    
    # Filter to CNN keys only (skip classifier/quantum which don't exist in classical)
    cnn_keys = {k: v for k, v in classical_state.items() if k.startswith('cnn.')}
    missing, unexpected = model.load_state_dict(cnn_keys, strict=False)
    
    print(f"  ✅ Loaded {len(cnn_keys)} CNN parameter tensors")
    print(f"  ℹ️  Skipped (not in classical): {len(missing)} keys (quantum + classifier)")
    if unexpected:
        print(f"  ⚠️  Unexpected keys: {unexpected}")
    
    # Report classical checkpoint performance
    if 'best_val_accuracy' in checkpoint:
        print(f"  📊 Classical checkpoint val accuracy: {checkpoint['best_val_accuracy']:.4f}")
    
    # Optionally freeze CNN
    if args.freeze_cnn:
        print("\n🧊 Freezing CNN backbone (only quantum + classifier will train)")
        for name, param in model.named_parameters():
            if name.startswith('cnn.'):
                param.requires_grad = False
    else:
        print("\n🔓 CNN backbone unfrozen (all parameters will train with lower CNN LR)")
    
    params = count_parameters(model)
    print(f"Model parameters: {params}")
    
    # Train
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_on_test_set(
        model, test_loader, device, class_names,
        save_path=f'{config.output_dir}/test_results.json'
    )
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(
        model, train_loader, test_loader, config, device,
        save_dir=f'{config.output_dir}/feature_analysis'
    )
    
    print("\nFine-tuning complete!")
    return model, history, results


def main():
    """Main entry point."""
    args = parse_args()
    config = build_config(args)
    
    # Save config
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(f'{config.output_dir}/config.yaml')
    
    if args.mode == 'train':
        train_model(args, config)
    elif args.mode == 'finetune':
        finetune_model(args, config)
    elif args.mode == 'evaluate':
        evaluate_model(args, config)
    elif args.mode == 'explain':
        explain_prediction(args, config)
    elif args.mode == 'ablation':
        run_ablation(args, config)
    elif args.mode == 'xgboost':
        run_xgboost_comparison(args, config)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
