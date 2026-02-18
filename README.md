# Hybrid Quantum-Classical Machine Learning for Medical Imaging

A research-grade implementation of hybrid quantum-classical deep learning for brain cancer MRI classification with explainable AI (XAI).

## Overview

This project implements a complete pipeline combining:
- **CNN Feature Extraction**: Custom lightweight CNN compressing images to ≤10 features
- **Quantum Processing**: 10-qubit PQC with RY+RZ encoding, ring entanglement, and final rotation layer
- **Classical Baseline**: XGBoost classification for comparison
- **Explainable AI**: Grad-CAM, SHAP, and LIME for interpretability

## Architecture

```
MRI Image → CNN (Conv→Pool→GAP→Dense) → 10 features → Quantum Layer → Classifier → Prediction
                                              ↓
                                         XGBoost (baseline)
```

### Quantum Circuit
- **Encoding**: RY angle encoding (features normalized to [0, π] via sigmoid scaling)
- **Variational**: 2 layers of trainable RY + RZ rotations (full Bloch sphere coverage)
- **Entanglement**: Ring CNOT pattern (symmetric connectivity)
- **Final Rotation**: Trainable RY + RZ before measurement (optimal measurement basis)
- **Measurement**: Pauli-Z expectations on all qubits
- **Parameters**: 60 trainable (vs 20 with old RY-only design)

## Installation

```bash
cd qml_medical_imaging
pip install -r requirements.txt
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- PennyLane 0.33+
- XGBoost, SHAP, LIME
- OpenCV, PIL, NumPy, Matplotlib

## Project Structure

```
qml_medical_imaging/
├── configs/           # Configuration management
│   └── config.py      # Dataclass-based config
├── data/              # Data loading and preprocessing
│   ├── dataset.py     # MRI dataset class
│   ├── transforms.py  # Image augmentation
│   └── utils.py       # Split utilities
├── models/            # Model architectures
│   ├── cnn_feature_extractor.py
│   ├── hybrid_model.py
│   └── xgboost_classifier.py
├── quantum/           # Quantum components
│   ├── circuits.py    # PQC definitions
│   ├── quantum_layer.py # PyTorch integration
│   └── utils.py       # Encoding utilities
├── training/          # Training infrastructure
│   ├── trainer.py     # Training loop
│   ├── losses.py      # Loss functions
│   ├── metrics.py     # Evaluation metrics
│   └── logger.py      # Logging utilities
├── evaluation/        # Evaluation and analysis
│   ├── evaluate.py    # Model evaluation
│   ├── ablation.py    # Ablation studies
│   └── feature_importance.py
├── xai/               # Explainable AI
│   ├── gradcam.py     # Grad-CAM implementation
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   └── visualizations.py
├── main.py            # Training entry point
├── inference.py       # Inference script
└── requirements.txt
```

## Usage

### 1. Train Hybrid Model (CNN + Quantum)

```bash
python main.py --mode train \
    --model_type hybrid \
    --data_dir ./data/brain_tumor_dataset \
    --epochs 50 \
    --lr 1e-3 \
    --quantum_lr 1e-2 \
    --n_qubits 10 \
    --n_layers 2 \
    --entanglement ring \
    --batch_size 16 \
    --output_dir ./outputs
```

### 2. Train Classical-Only Baseline

```bash
python main.py --mode train \
    --model_type classical \
    --data_dir ./data/brain_tumor_dataset \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 16 \
    --output_dir ./outputs_classical
```

### 3. Evaluate a Trained Model

```bash
python main.py --mode evaluate \
    --model_type hybrid \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/brain_tumor_dataset \
    --n_qubits 10 \
    --n_layers 2 \
    --entanglement ring \
    --output_dir ./outputs
```

> **Important:** `--n_qubits`, `--n_layers`, and `--entanglement` must match training config, otherwise checkpoint loading will fail with an architecture mismatch error.

### 4. Generate XAI Explanations

```bash
python main.py --mode explain \
    --checkpoint ./checkpoints/best_model.pt \
    --image ./sample_image.jpg
```

### 5. Run Ablation Study (Hybrid vs Classical)

```bash
python main.py --mode ablation \
    --data_dir ./data/brain_tumor_dataset \
    --n_qubits 10 \
    --n_layers 2 \
    --entanglement ring \
    --output_dir ./outputs/ablation
```

### 6. Feature Bottleneck Experiment

```bash
python experiments/feature_bottleneck.py \
    --data_dir ./data/brain_tumor_dataset \
    --dimensions 10 32 64 \
    --epochs 30 \
    --batch_size 16 \
    --output_dir ./outputs/feature_bottleneck
```

### 7. Inference

```bash
# Single image
python inference.py --checkpoint ./checkpoints/best_model.pt \
    --image /path/to/image.jpg --explain

# Batch inference
python inference.py --checkpoint ./checkpoints/best_model.pt \
    --image_dir ./test_images --explain
```

## Configuration

All hyperparameters are managed via `configs/config.py`:

```python
from configs.config import Config

config = Config()
config.data.batch_size = 32
config.quantum.n_layers = 3
config.training.learning_rate = 1e-3
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.num_features` | 10 | CNN output dimension (max 10 for quantum) |
| `quantum.n_qubits` | 10 | Number of qubits |
| `quantum.n_layers` | 2 | PQC variational layers |
| `quantum.entanglement` | 'ring' | CNOT pattern ('linear', 'ring', 'full') |
| `training.num_epochs` | 50 | Training epochs |
| `training.learning_rate` | 1e-3 | Classical learning rate |
| `training.quantum_learning_rate` | 1e-2 | Quantum parameters LR |

## Dataset

Expects data organized as:
```
data/brain_mri/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```

## Explainable AI

### Grad-CAM
Visualizes CNN attention regions highlighting important areas for classification.

### SHAP
Feature-level importance analysis for both CNN and quantum features.

### LIME
Local interpretable explanations showing which image regions contribute to predictions.

## Research Notes

1. **Expressive PQC**: RY+RZ rotations access full Bloch sphere (vs RY-only = real amplitudes)
2. **Ring Entanglement**: Symmetric qubit connectivity, breaks boundary effects of linear chains
3. **Final Rotation Layer**: Aligns measurement basis with learned representation
4. **Sigmoid Normalization**: Deterministic feature→angle mapping, consistent across train/inference
5. **Separate Learning Rates**: Quantum params use 10× higher LR (1e-2 vs 1e-3)
6. **Feature Compression**: ≤10 features enables direct qubit mapping
7. **Dynamic pin_memory**: Auto-disabled on MPS (Apple Silicon) to avoid PyTorch warnings

## Citation

```bibtex
@article{hybrid_qml_medical,
  title={Hybrid Quantum-Classical Machine Learning for Medical Imaging},
  year={2024}
}
```

## License

MIT License
