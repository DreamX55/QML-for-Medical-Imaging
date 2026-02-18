"""
Training Logger for Experiment Tracking.

This module provides logging utilities including:
- Console logging with formatting
- File logging for reproducibility
- TensorBoard integration
- History saving and loading

Design Decisions:
- Structured logging for easy parsing
- TensorBoard for visualization
- JSON history for analysis
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class TrainingLogger:
    """
    Comprehensive training logger.
    
    Provides:
    - Console output with formatting
    - File logging (log.txt)
    - TensorBoard integration
    - History management
    
    Attributes:
        log_dir: Directory for logs
        experiment_name: Name of the experiment
        log_file: Path to text log file
        tb_writer: TensorBoard SummaryWriter (if available)
    """
    
    def __init__(
        self,
        log_dir: str = './outputs/logs',
        experiment_name: str = 'experiment'
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for this experiment
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = self.log_dir / f'{experiment_name}_{timestamp}'
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log file
        self.log_file = self.exp_dir / 'training.log'
        
        # Setup TensorBoard
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.exp_dir / 'tensorboard')
            self._log("TensorBoard logging enabled")
        except ImportError:
            self._log("TensorBoard not available, skipping")
        
        self._log(f"Experiment: {experiment_name}")
        self._log(f"Log directory: {self.exp_dir}")
    
    def _log(self, message: str) -> None:
        """Write message to console and log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        
        print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def log_config(self, config: dict) -> None:
        """
        Log configuration to file.
        
        Args:
            config: Configuration dictionary
        """
        config_path = self.exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self._log(f"Configuration saved to {config_path}")
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        epoch_time: float
    ) -> None:
        """
        Log metrics for one epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            learning_rate: Current learning rate
            epoch_time: Time taken for epoch
        """
        # Log to TensorBoard
        if self.tb_writer is not None:
            for name, value in train_metrics.items():
                self.tb_writer.add_scalar(f'train/{name}', value, epoch)
            
            for name, value in val_metrics.items():
                self.tb_writer.add_scalar(f'val/{name}', value, epoch)
            
            self.tb_writer.add_scalar('learning_rate', learning_rate, epoch)
        
        # Format log message
        train_str = ' | '.join([f'{k}: {v:.4f}' for k, v in train_metrics.items()])
        val_str = ' | '.join([f'{k}: {v:.4f}' for k, v in val_metrics.items()])
        
        self._log(f"Epoch {epoch+1}")
        self._log(f"  Train: {train_str}")
        self._log(f"  Val: {val_str}")
        self._log(f"  LR: {learning_rate:.6f} | Time: {epoch_time:.1f}s")
    
    def log_message(self, message: str) -> None:
        """Log an arbitrary message."""
        self._log(message)
    
    def save_history(self, history: Dict[str, List[float]]) -> None:
        """
        Save training history to JSON.
        
        Args:
            history: Dictionary of metric histories
        """
        history_path = self.exp_dir / 'history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        history_clean = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                history_clean[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.floating):
                    history_clean[key] = [float(v) for v in value]
                else:
                    history_clean[key] = value
            else:
                history_clean[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_clean, f, indent=2)
        
        self._log(f"History saved to {history_path}")
    
    def load_history(self, path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Load training history from JSON.
        
        Args:
            path: Path to history file (uses default if None)
            
        Returns:
            History dictionary
        """
        if path is None:
            path = self.exp_dir / 'history.json'
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def log_model_summary(self, model, input_shape: tuple = (1, 3, 224, 224)) -> None:
        """
        Log model architecture summary.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape for summary
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self._log("Model Summary:")
        self._log(f"  Total parameters: {total_params:,}")
        self._log(f"  Trainable parameters: {trainable_params:,}")
        self._log(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Try to use torchinfo if available
        try:
            from torchinfo import summary
            summary_str = summary(
                model, 
                input_shape, 
                verbose=0,
                col_names=['input_size', 'output_size', 'num_params']
            )
            
            summary_path = self.exp_dir / 'model_summary.txt'
            with open(summary_path, 'w') as f:
                f.write(str(summary_str))
            
            self._log(f"  Full summary saved to {summary_path}")
        except ImportError:
            pass
    
    def log_evaluation_results(
        self,
        results: Dict[str, float],
        phase: str = 'test'
    ) -> None:
        """
        Log evaluation results.
        
        Args:
            results: Dictionary of evaluation metrics
            phase: Phase name ('test', 'val', etc.)
        """
        self._log(f"\n{phase.upper()} Results:")
        for name, value in results.items():
            self._log(f"  {name}: {value:.4f}")
        
        # Save to JSON
        results_path = self.exp_dir / f'{phase}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def close(self) -> None:
        """Close logger and release resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        self._log("Training logger closed")


def create_experiment_logger(
    base_dir: str,
    experiment_name: str,
    config: Optional[dict] = None
) -> TrainingLogger:
    """
    Factory function to create and setup logger.
    
    Args:
        base_dir: Base directory for logs
        experiment_name: Experiment name
        config: Optional configuration to log
        
    Returns:
        Configured TrainingLogger instance
    """
    logger = TrainingLogger(base_dir, experiment_name)
    
    if config is not None:
        logger.log_config(config)
    
    return logger
