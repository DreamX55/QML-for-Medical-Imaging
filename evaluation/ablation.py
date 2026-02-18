"""
Ablation Studies for Hybrid Quantum-Classical Models.

Compares performance with/without quantum layer
and different architecture variations.

FIXED: Now loads pre-trained checkpoints instead of retraining from scratch.
The original version retrained both models from random initialization,
causing the hybrid model to show ~36% accuracy (quantum circuits need
many epochs to converge) instead of the actual ~81%.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.hybrid_model import HybridQuantumClassifier, create_model
from training.trainer import Trainer
from .evaluate import evaluate_model


def debug_forward_pass(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_name: str = "MODEL"
) -> None:
    """
    Run a single batch through the model and print tensor shapes
    at every stage to verify the forward pass is correct.
    
    Args:
        model: Model to debug
        dataloader: Data loader (uses first batch only)
        device: Device to run on
        model_name: Label for print statements
    """
    model.eval()
    model.to(device)
    
    # Get a single batch
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    print(f"\n{'='*60}")
    print(f"DEBUG FORWARD PASS: {model_name}")
    print(f"{'='*60}")
    print(f"  Input shape: {images.shape}")
    
    with torch.no_grad():
        # Step 1: CNN features
        cnn_features = model.cnn(images)
        print(f"  CNN output shape: {cnn_features.shape}")
        print(f"  CNN output range: [{cnn_features.min():.4f}, {cnn_features.max():.4f}]")
        
        # Step 2: Quantum layer (if present)
        if hasattr(model, 'use_quantum') and model.use_quantum and model.quantum_layer is not None:
            quantum_features = model.quantum_layer(cnn_features)
            print(f"  Quantum output shape: {quantum_features.shape}")
            print(f"  Quantum output range: [{quantum_features.min():.4f}, {quantum_features.max():.4f}]")
            features = quantum_features
        else:
            print(f"  Quantum layer: DISABLED (classical-only mode)")
            features = cnn_features
        
        # Step 3: Classifier head
        logits = model.classifier(features)
        print(f"  Final logits shape: {logits.shape}")
        print(f"  Final logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        
        # Step 4: Softmax probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Sample predictions: {preds[:8].tolist()}")
        print(f"  Sample labels:      {labels[:8].tolist()}")
        
        # Verify full forward pass matches step-by-step
        full_output = model(images)
        match = torch.allclose(full_output, logits, atol=1e-6)
        print(f"  Full forward pass matches step-by-step: {match}")
        if not match:
            diff = (full_output - logits).abs().max().item()
            print(f"  WARNING: Max difference = {diff:.8f}")
    
    print(f"{'='*60}\n")


def run_ablation_study(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: Optional[List[str]] = None,
    device: str = 'cpu',
    save_dir: str = './outputs/ablation',
    hybrid_checkpoint: Optional[str] = None,
    classical_checkpoint: Optional[str] = None,
) -> Dict:
    """
    Run ablation study comparing hybrid vs classical.
    
    Loads pre-trained checkpoints for evaluation instead of retraining.
    Falls back to training from scratch only if no checkpoint is found.
    
    Args:
        config: Configuration object
        train_loader: Training data
        val_loader: Validation data
        test_loader: Test data
        class_names: Class names
        device: Device to use
        save_dir: Directory to save results
        hybrid_checkpoint: Path to hybrid model checkpoint (auto-detected if None)
        classical_checkpoint: Path to classical model checkpoint (auto-detected if None)
        
    Returns:
        Ablation results dictionary
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Auto-detect checkpoint paths
    checkpoint_dir = Path(config.training.checkpoint_dir)
    if hybrid_checkpoint is None:
        hybrid_checkpoint = checkpoint_dir / 'best_model.pt'
    else:
        hybrid_checkpoint = Path(hybrid_checkpoint)
    
    if classical_checkpoint is None:
        classical_checkpoint = checkpoint_dir / 'best_model_classical.pt'
    else:
        classical_checkpoint = Path(classical_checkpoint)
    
    # =========================================================
    # 1. Hybrid model (with quantum) — LOAD from checkpoint
    # =========================================================
    print("\n" + "="*50)
    print("Evaluating HYBRID model (with quantum)")
    print("="*50)
    
    hybrid_model = create_model(config, model_type='hybrid')
    
    if hybrid_checkpoint.exists():
        print(f"Loading hybrid checkpoint: {hybrid_checkpoint}")
        checkpoint = torch.load(hybrid_checkpoint, map_location=device)
        hybrid_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint best val accuracy: {checkpoint.get('best_val_accuracy', 'N/A')}")
        hybrid_history = checkpoint.get('history', {})
    else:
        print(f"WARNING: No hybrid checkpoint found at {hybrid_checkpoint}")
        print("Training hybrid model from scratch...")
        hybrid_trainer = Trainer(hybrid_model, config, device)
        hybrid_history = hybrid_trainer.train(
            train_loader, val_loader, 
            num_epochs=config.training.num_epochs
        )
    
    hybrid_model.to(device)
    hybrid_model.eval()
    
    # Debug: verify forward pass shapes
    debug_forward_pass(hybrid_model, test_loader, device, "HYBRID")
    
    # Evaluate
    hybrid_results = evaluate_model(hybrid_model, test_loader, device, class_names)
    results['hybrid'] = {
        'metrics': hybrid_results['metrics'],
        'history': {
            k: [float(v) for v in vals] 
            for k, vals in hybrid_history.items()
        } if hybrid_history else {}
    }
    
    print(f"Hybrid accuracy: {hybrid_results['metrics']['accuracy']:.4f}")
    
    # =========================================================
    # 2. Classical model (without quantum) — LOAD or train
    # =========================================================
    print("\n" + "="*50)
    print("Evaluating CLASSICAL model (without quantum)")
    print("="*50)
    
    classical_model = create_model(config, model_type='classical')
    
    if classical_checkpoint.exists():
        print(f"Loading classical checkpoint: {classical_checkpoint}")
        checkpoint = torch.load(classical_checkpoint, map_location=device)
        classical_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Checkpoint best val accuracy: {checkpoint.get('best_val_accuracy', 'N/A')}")
        classical_history = checkpoint.get('history', {})
    else:
        print(f"No classical checkpoint found at {classical_checkpoint}")
        print("Training classical model from scratch...")
        classical_trainer = Trainer(classical_model, config, device)
        classical_history = classical_trainer.train(
            train_loader, val_loader,
            num_epochs=config.training.num_epochs
        )
    
    classical_model.to(device)
    classical_model.eval()
    
    # Debug: verify forward pass shapes
    debug_forward_pass(classical_model, test_loader, device, "CLASSICAL")
    
    # Evaluate
    classical_results = evaluate_model(classical_model, test_loader, device, class_names)
    results['classical'] = {
        'metrics': classical_results['metrics'],
        'history': {
            k: [float(v) for v in vals] 
            for k, vals in classical_history.items()
        } if classical_history else {}
    }
    
    print(f"Classical accuracy: {classical_results['metrics']['accuracy']:.4f}")
    
    # =========================================================
    # 3. Compute improvements
    # =========================================================
    results['quantum_improvement'] = {
        metric: results['hybrid']['metrics'][metric] - results['classical']['metrics'][metric]
        for metric in results['hybrid']['metrics']
    }
    
    # Save results
    with open(f'{save_dir}/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("ABLATION STUDY RESULTS")
    print("="*50)
    print(f"{'Metric':<15} {'Hybrid':<10} {'Classical':<10} {'Diff':<10}")
    print("-"*50)
    for metric in ['accuracy', 'f1', 'auc']:
        if metric in results['hybrid']['metrics']:
            h = results['hybrid']['metrics'][metric]
            c = results['classical']['metrics'][metric]
            d = results['quantum_improvement'][metric]
            print(f"{metric:<15} {h:.4f}     {c:.4f}      {d:+.4f}")
    
    return results


def compare_models(
    models: Dict[str, nn.Module],
    test_loader: DataLoader,
    device: str = 'cpu',
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compare multiple models on test set.
    
    Args:
        models: Dictionary of {name: model}
        test_loader: Test data
        device: Device to use
        class_names: Class names
        
    Returns:
        Comparison results
    """
    results = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Debug each model's forward pass
        debug_forward_pass(model, test_loader, device, name.upper())
        
        model_results = evaluate_model(model, test_loader, device, class_names)
        results[name] = model_results['metrics']
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    metrics = list(list(results.values())[0].keys())
    header = f"{'Model':<20}" + "".join([f"{m:<12}" for m in metrics])
    print(header)
    print("-"*60)
    
    for name, metrics_dict in results.items():
        row = f"{name:<20}" + "".join([f"{v:.4f}      " for v in metrics_dict.values()])
        print(row)
    
    return results


def analyze_circuit_depth_impact(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    depths: List[int] = [1, 2, 3, 4],
    device: str = 'cpu'
) -> Dict:
    """
    Analyze impact of quantum circuit depth.
    
    Args:
        config: Base configuration
        train_loader: Training data
        val_loader: Validation data
        test_loader: Test data
        depths: Circuit depths to test
        device: Device to use
        
    Returns:
        Results for each depth
    """
    results = {}
    
    for depth in depths:
        print(f"\nTesting circuit depth: {depth}")
        
        # Modify config
        test_config = config
        test_config.quantum.n_layers = depth
        
        # Train and evaluate
        model = create_model(test_config, model_type='hybrid')
        trainer = Trainer(model, test_config, device)
        trainer.train(train_loader, val_loader, num_epochs=test_config.training.num_epochs // 2)
        
        # Debug forward pass
        debug_forward_pass(model, test_loader, device, f"DEPTH_{depth}")
        
        eval_results = evaluate_model(model, test_loader, device)
        results[f'depth_{depth}'] = eval_results['metrics']
    
    return results
