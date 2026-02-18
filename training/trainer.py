"""
Training Loop for Hybrid Quantum-Classical Models.

This module provides the main training infrastructure including:
- Training and validation epoch functions
- Full Trainer class with checkpointing and early stopping
- Support for separate learning rates for quantum/classical parameters

Design Decisions:
- Separate optimizer groups for quantum and classical parameters
- Gradient clipping for stable hybrid training
- Comprehensive logging and checkpointing
- Early stopping with patience
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import Config, TrainingConfig
from .losses import get_loss_function
from .metrics import compute_metrics, MetricsTracker
from .logger import TrainingLogger


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    gradient_clip: float = 1.0,
    log_interval: int = 10
) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        gradient_clip: Maximum gradient norm
        log_interval: Log every N batches
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Track loss and predictions
        total_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, Dict[str, float]]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(
        all_labels, 
        all_predictions, 
        np.array(all_probs)
    )
    metrics['loss'] = avg_loss
    
    return avg_loss, metrics


class Trainer:
    """
    Full training pipeline for hybrid quantum-classical models.
    
    Features:
    - Separate learning rates for quantum and classical parameters
    - Learning rate scheduling
    - Early stopping
    - Checkpointing (best and periodic)
    - Comprehensive logging
    
    Attributes:
        model: Model to train
        config: Training configuration
        device: Training device
        logger: Training logger instance
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Config,
        device: Optional[str] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Full configuration object
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.train_config = config.training
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model.to(self.device)
        print(f"Training on device: {self.device}")
        
        # Setup loss function
        self.criterion = get_loss_function(config.model.num_classes)
        
        # Setup optimizer with separate parameter groups
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup logger
        self.logger = TrainingLogger(
            log_dir=self.train_config.log_dir,
            experiment_name=config.experiment_name
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'learning_rate': []
        }
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup optimizer with separate learning rates for CNN and quantum parameters.
        
        Parameter groups:
          - 'cnn': CNN backbone + compression layers (LR = training.learning_rate)
          - 'classifier': Dense classification head (LR = training.learning_rate)
          - 'quantum': Quantum layer weights (LR = training.quantum_learning_rate)
        
        Quantum parameters benefit from a higher LR (default 10x) because:
          - Quantum landscape has a narrower useful gradient region
          - Quantum params (rotation angles) are periodic and bounded
          - Smaller absolute gradients from quantum circuits need amplification
        """
        # Separate parameters into groups by component
        cnn_params = []
        classifier_params = []
        quantum_params = []
        frozen_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                frozen_params.append((name, param))
                continue
            
            if 'quantum' in name.lower() or name == 'weights':
                # Quantum layer weights (includes the nn.Parameter named 'weights')
                quantum_params.append(param)
            elif 'classifier' in name.lower():
                classifier_params.append(param)
            else:
                # CNN backbone + compression
                cnn_params.append(param)
        
        # ── Parameter audit ──
        print(f"\n{'='*55}")
        print(f"  OPTIMIZER PARAMETER AUDIT")
        print(f"{'='*55}")
        
        cnn_count = sum(p.numel() for p in cnn_params)
        classifier_count = sum(p.numel() for p in classifier_params)
        quantum_count = sum(p.numel() for p in quantum_params)
        frozen_count = sum(p.numel() for p in (p for _, p in frozen_params))
        total_model = sum(p.numel() for p in self.model.parameters())
        total_trainable = cnn_count + classifier_count + quantum_count
        
        print(f"  CNN backbone + compression:  {cnn_count:>8,} params  (lr={self.train_config.learning_rate})")
        print(f"  Classifier head:             {classifier_count:>8,} params  (lr={self.train_config.learning_rate})")
        print(f"  Quantum layer:               {quantum_count:>8,} params  (lr={self.train_config.quantum_learning_rate})")
        print(f"  {'─'*51}")
        print(f"  Total trainable:             {total_trainable:>8,}")
        
        # Warn about frozen parameters
        if frozen_params:
            print(f"\n  ⚠️  FROZEN PARAMETERS ({frozen_count:,} params):")
            for name, param in frozen_params:
                print(f"      {name}: {param.shape} ({param.numel()} params)")
            print(f"  These parameters will NOT be updated during training!")
        
        # Verify all parameters are accounted for
        if total_trainable + frozen_count != total_model:
            missing = total_model - total_trainable - frozen_count
            print(f"\n  🔴 ERROR: {missing} parameters are NOT in any optimizer group!")
            # Find the missing ones
            optimizer_param_ids = set(
                id(p) for p in cnn_params + classifier_params + quantum_params
            ) | set(id(p) for _, p in frozen_params)
            for name, param in self.model.named_parameters():
                if id(param) not in optimizer_param_ids:
                    print(f"      MISSING: {name}: {param.shape}")
        else:
            print(f"\n  ✅ All {total_model:,} model parameters accounted for")
        
        print(f"{'='*55}\n")
        
        # Build parameter groups (only include non-empty groups)
        param_groups = []
        
        if cnn_params:
            param_groups.append({
                'params': cnn_params,
                'lr': self.train_config.learning_rate,
                'name': 'cnn'
            })
        
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': self.train_config.learning_rate,
                'name': 'classifier'
            })
        
        if quantum_params:
            param_groups.append({
                'params': quantum_params,
                'lr': self.train_config.quantum_learning_rate,
                'name': 'quantum'
            })
        
        if not param_groups:
            raise RuntimeError("No trainable parameters found! Check model initialization.")
        
        # Create optimizer
        if self.train_config.optimizer.lower() == 'adam':
            optimizer = Adam(
                param_groups,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == 'adamw':
            optimizer = AdamW(
                param_groups,
                weight_decay=self.train_config.weight_decay
            )
        elif self.train_config.optimizer.lower() == 'sgd':
            optimizer = SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.train_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config.optimizer}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if not self.train_config.use_scheduler:
            return None
        
        if self.train_config.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.num_epochs,
                eta_min=1e-6
            )
        elif self.train_config.scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=10,
                gamma=self.train_config.scheduler_factor
            )
        elif self.train_config.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.train_config.scheduler_patience,
                factor=self.train_config.scheduler_factor
            )
        else:
            return None
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            
        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.train_config.num_epochs
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print()
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_loss, train_metrics = train_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                gradient_clip=self.train_config.gradient_clip_value,
                log_interval=self.train_config.log_interval
            )
            
            # Validation
            val_loss, val_metrics = validate_epoch(
                self.model,
                val_loader,
                self.criterion,
                self.device
            )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(current_lr)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.log_epoch(
                epoch, 
                train_metrics, 
                val_metrics, 
                current_lr,
                epoch_time
            )
            
            # Print progress
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                if self.train_config.save_best_only:
                    self._save_checkpoint('best_model.pt', is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if (self.train_config.use_early_stopping and 
                self.patience_counter >= self.train_config.early_stopping_patience):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        
        # Save final model
        self._save_checkpoint('final_model.pt', is_best=False)
        
        # Save history
        self.logger.save_history(self.history)
        
        return self.history
    
    def _save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.train_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history,
            'config': self.config._to_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            print(f"  Saved best model to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
