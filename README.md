# Hybrid Quantum-Classical MRI Classifier

## Project Overview
This repository implements a Hybrid Quantum-Classical MRI classifier using:
- ResNet18 transfer learning
- Frozen backbone (quantum transfer learning)
- 10-qubit PQC
- Classical-Quantum ensemble
- Explainable AI (GradCAM + SHAP + LIME)

## Current Production Architecture
The hybrid pipeline operates as follows:

```text
MRI → ResNet18 (frozen) → 512 features
  ├──→ Branch A: Classical linear classifier
  └──→ Branch B: Dense → BatchNorm → 10 features → Quantum circuit → classifier
       └──→ Ensemble (0.5 classical + 0.5 quantum)
```

**Current Performance:** ~92–93% accuracy, ~0.98 AUC.

## Why Quantum Is Used
The quantum layer models complex nonlinear interactions in latent feature space and acts as an ensemble booster rather than a replacement for classical ML.

## High-Performance Training on Colab
- **Main training notebook:** `notebooks/colab_high_performance.ipynb`
- **Target environment:** Designed for GPU runtime

**Recommended environment steps:**
1. Install requirements
2. Upload dataset
3. Run notebook top-to-bottom

## Classical Baseline & XAI Comparison
This repository includes:
- Strong classical baseline model
- Hybrid vs Classical comparison experiments
- Explainability comparison across models

## Repository Structure
```text
.
├── configs/                 # Configuration and hyperparameter settings
├── data/                    # Data loading and preprocessing
├── evaluation/              # Evaluation and analysis
├── experiments/             # Architecture experiments (Feature Bottleneck)
├── models/                  # CNN feature extractors and hybrid architectures
├── notebooks/               # Interactive training and evaluation notebooks
│   └── colab_high_performance.ipynb
├── quantum/                 # Quantum layer implementations and circuits
├── training/                # Training loop, losses, and metrics
├── xai/                     # Explainable AI (GradCAM, SHAP, LIME)
├── main.py                  # CLI entry point for local execution
├── inference.py             # Inference script
├── architecture_docs.md     # In-depth design and theory documentation
├── requirements.txt         # Project dependencies
└── README.md                # This document
```

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

**1. Train Hybrid Model:**
```bash
python main.py --mode train --model_type hybrid --data_dir ./data/brain_tumor_dataset --n_qubits 10 --n_layers 2 --entanglement ring
```

**2. Train Classical Baseline:**
```bash
python main.py --mode train --model_type classical --data_dir ./data/brain_tumor_dataset
```

**3. Evaluate and Generate XAI Explanations:**
```bash
python main.py --mode evaluate --model_type hybrid --checkpoint ./checkpoints/best_model.pt --data_dir ./data/brain_tumor_dataset
python main.py --mode explain --checkpoint ./checkpoints/best_model.pt --image ./sample_image.jpg
```

## License
MIT License
