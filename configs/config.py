"""
Central Configuration Management for Hybrid Quantum-Classical ML Pipeline.

This module provides a comprehensive, type-safe configuration system using
dataclasses. All hyperparameters, paths, and architectural choices are
centralized here for reproducibility.

Design Decisions:
- Dataclasses provide type safety and IDE autocompletion
- Nested configs for logical grouping (data, model, quantum, training, xai)
- Default values chosen for simulator-friendly quantum experiments
- Paths are configurable to support different environments
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
import json


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Dataset path - should contain class-organized subfolders
    data_dir: str = "./data/brain_mri"
    
    # Image dimensions after preprocessing
    image_size: Tuple[int, int] = (224, 224)
    
    # Number of channels (1 for grayscale MRI, 3 for RGB)
    num_channels: int = 3
    
    # Train/validation/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data loading parameters
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    
    # Augmentation settings
    use_augmentation: bool = True
    random_rotation_degrees: int = 15
    random_horizontal_flip: bool = True
    random_vertical_flip: bool = False
    
    # Normalization (ImageNet defaults, adjust for MRI if needed)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Random seed for reproducible splits
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for CNN feature extractor."""
    
    # Number of output features from CNN (MUST be ≤10 for quantum encoding)
    # This is the compression dimension that feeds into the quantum circuit
    num_features: int = 10
    
    # CNN architecture settings
    # Using a custom lightweight CNN for interpretability
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Kernel size for convolutions
    kernel_size: int = 3
    
    # Pooling settings
    pool_size: int = 2
    
    # Dropout for regularization
    dropout_rate: float = 0.3
    
    # Use batch normalization
    use_batch_norm: bool = True
    
    # Pretrained backbone (None for custom CNN, or 'resnet18', 'efficientnet_b0')
    # For research, we use custom to have full control
    backbone: Optional[str] = None
    
    # Number of classes (will be set from dataset)
    num_classes: int = 3


@dataclass
class QuantumConfig:
    """Configuration for quantum circuit."""
    
    # Number of qubits (MUST match num_features from ModelConfig)
    n_qubits: int = 10
    
    # Number of variational layers in the PQC
    # Shallow depth to avoid barren plateaus and enable simulator execution
    n_layers: int = 2
    
    # Entanglement pattern: 'linear', 'ring', 'full'
    # Linear: each qubit connected to next (n-1 CNOTs)
    # Ring: linear + last-to-first connection (symmetric connectivity)
    # Full: all-to-all (expensive, not recommended)
    entanglement: str = "ring"
    
    # Quantum device/simulator
    # 'default.qubit' for CPU simulation
    # 'lightning.qubit' for faster simulation
    device: str = "default.qubit"
    
    # Whether to use parameter-shift rule (True) or backprop (False)
    # Parameter-shift is hardware-compatible but slower
    use_parameter_shift: bool = False
    
    # Feature encoding range
    # Features are normalized to this range before RY encoding
    encoding_range: Tuple[float, float] = (0.0, 3.14159)  # [0, π]
    
    # Number of measurement shots (None for analytic simulation)
    shots: Optional[int] = None
    
    # Output dimension from quantum circuit
    # This is the number of expectation values measured
    n_outputs: int = 10


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Number of training epochs
    num_epochs: int = 50
    
    # Learning rate
    learning_rate: float = 1e-3
    
    # Separate learning rate for quantum parameters (often needs different scale)
    quantum_learning_rate: float = 1e-2
    
    # Quantum Transfer Learning: Freeze backbone weights
    freeze_backbone: bool = False
    
    # Optimizer: 'adam', 'adamw', 'sgd'
    optimizer: str = "adam"
    
    # Weight decay for regularization
    weight_decay: float = 1e-4
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 'step', 'cosine', 'plateau'
    scheduler_patience: int = 5  # For plateau scheduler
    scheduler_factor: float = 0.5  # For step/plateau
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Gradient clipping (important for hybrid training stability)
    gradient_clip_value: float = 1.0
    
    # Mixed precision training (not recommended for quantum simulation)
    use_mixed_precision: bool = False
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "./outputs/checkpoints"
    
    # Logging
    log_dir: str = "./outputs/logs"
    log_interval: int = 10  # Log every N batches
    
    # Device
    device: str = "cuda"  # Will fall back to CPU if CUDA unavailable
    
    # Random seed
    seed: int = 42


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost baseline."""
    
    # Number of boosting rounds
    n_estimators: int = 100
    
    # Maximum tree depth
    max_depth: int = 6
    
    # Learning rate
    learning_rate: float = 0.1
    
    # Subsample ratio
    subsample: float = 0.8
    
    # Column subsample ratio
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    
    # Objective for multi-class
    objective: str = "multi:softprob"
    
    # Random seed
    seed: int = 42
    
    # Number of parallel threads
    n_jobs: int = -1


@dataclass
class XAIConfig:
    """Configuration for Explainable AI modules."""
    
    # Grad-CAM settings
    gradcam_target_layer: str = "features.conv4"  # Layer to visualize
    
    # SHAP settings
    shap_background_samples: int = 100  # Number of background samples
    shap_max_display: int = 10  # Max features to display
    
    # LIME settings
    lime_num_samples: int = 1000  # Perturbation samples
    lime_num_features: int = 10  # Top features to show
    lime_kernel_width: float = 0.25
    
    # Visualization settings
    colormap: str = "jet"  # For heatmaps
    alpha_overlay: float = 0.4  # Transparency for overlays
    figure_size: Tuple[int, int] = (12, 8)
    save_format: str = "png"
    dpi: int = 150
    
    # Output directory for explanations
    output_dir: str = "./outputs/explanations"


@dataclass
class Config:
    """Master configuration combining all sub-configurations."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    
    # Project-level settings
    project_name: str = "hybrid_qml_medical_imaging"
    experiment_name: str = "default"
    
    # Output directories
    output_dir: str = "./outputs"
    
    # Random seed (master seed, propagated to sub-configs)
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration consistency after initialization."""
        # Ensure quantum qubits match CNN output features
        if self.model.num_features != self.quantum.n_qubits:
            raise ValueError(
                f"CNN output features ({self.model.num_features}) must match "
                f"quantum qubits ({self.quantum.n_qubits})"
            )
        
        # Ensure feature count ≤ 10 for quantum encoding
        if self.model.num_features > 10:
            raise ValueError(
                f"CNN output features ({self.model.num_features}) must be ≤10 "
                "for quantum encoding"
            )
        
        # Propagate master seed
        self.data.seed = self.seed
        self.training.seed = self.seed
        self.xgboost.seed = self.seed
        
        # Create output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.xai.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self._to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls._from_dict(config_dict)
    
    def _to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'quantum': self.quantum.__dict__,
            'training': self.training.__dict__,
            'xgboost': self.xgboost.__dict__,
            'xai': self.xai.__dict__,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'seed': self.seed,
        }
    
    @classmethod
    def _from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            quantum=QuantumConfig(**config_dict.get('quantum', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            xgboost=XGBoostConfig(**config_dict.get('xgboost', {})),
            xai=XAIConfig(**config_dict.get('xai', {})),
            project_name=config_dict.get('project_name', 'hybrid_qml_medical_imaging'),
            experiment_name=config_dict.get('experiment_name', 'default'),
            output_dir=config_dict.get('output_dir', './outputs'),
            seed=config_dict.get('seed', 42),
        )
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        return json.dumps(self._to_dict(), indent=2, default=str)
