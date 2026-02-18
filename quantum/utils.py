"""
Quantum Utilities for Feature Encoding and Processing.

This module provides utility functions for:
- Feature normalization for quantum encoding
- Angle encoding transformations
- Quantum device management
- Circuit analysis utilities

Design Decisions:
- Normalization to [0, π] for RY encoding (natural range)
- Uses sigmoid-based scaling for stable, input-independent normalization
- Support for alternative encoding schemes
- Device-agnostic utilities

IMPORTANT — WHY PROPER SCALING MATTERS:
  Quantum gates (RY) use rotation angles. If the scaling is inconsistent:
  1. The same feature value produces DIFFERENT angles in different batches,
     because per-sample min/max normalization depends on batch composition.
  2. During training large batches spread the range; during inference a
     single sample maps min→0 and max→π regardless of absolute value.
  3. This train/inference mismatch directly degrades accuracy.
  4. Compressed ranges (all angles near π/2) waste the gate's dynamic
     range, while extreme angles saturate the cosine, killing gradients.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union
import pennylane as qml

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import QuantumConfig


def normalize_features(
    features: torch.Tensor,
    out_min: float = 0.0,
    out_max: float = np.pi,
    in_min: Optional[float] = None,
    in_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Normalize features to a specified range for quantum encoding.
    
    Uses a sigmoid-based mapping that is:
    - STABLE: The same input always produces the same angle, regardless
      of what other values are in the batch.
    - SMOOTH: No hard clipping; extreme values asymptotically approach
      the bounds without ever being silently truncated.
    - CONSISTENT: Identical behavior during training and inference.
    
    Why NOT per-sample min/max (the old approach):
      If a batch has features [0.1, 0.5, 0.9], min/max maps them to
      [0, π/2, π]. But a single-sample batch [0.5] maps it to [0] (or
      [π]), completely changing its angle. This breaks inference.
    
    Args:
        features: Input tensor of any shape
        out_min: Minimum of output range (default: 0)
        out_max: Maximum of output range (default: π)
        in_min: IGNORED (kept for API compat). Use normalize_features_minmax()
                if you truly need data-dependent scaling.
        in_max: IGNORED (kept for API compat).
        
    Returns:
        Normalized tensor with values in (out_min, out_max)
        Center of range corresponds to input = 0.
    """
    # Sigmoid maps (-∞, +∞) → (0, 1) smoothly and deterministically.
    # Same input → same output, regardless of batch composition.
    normalized = torch.sigmoid(features)  # ∈ (0, 1)
    
    # Scale to target range
    scaled = normalized * (out_max - out_min) + out_min
    
    return scaled


def normalize_features_minmax(
    features: torch.Tensor,
    out_min: float = 0.0,
    out_max: float = np.pi,
    in_min: Optional[float] = None,
    in_max: Optional[float] = None,
) -> torch.Tensor:
    """
    [DEPRECATED — prefer normalize_features()]
    
    Min-max normalization to a specified range.
    
    WARNING: This uses per-tensor min/max, so the same feature value
    will produce DIFFERENT rotation angles depending on what other
    values are in the batch. This causes train/inference mismatch.
    
    Only use this if you have fixed, known in_min / in_max values
    that are constant across all data.
    
    Args:
        features: Input tensor
        out_min: Output range minimum
        out_max: Output range maximum
        in_min: Input range minimum (auto-detected if None — UNSTABLE)
        in_max: Input range maximum (auto-detected if None — UNSTABLE)
    """
    if in_min is None:
        in_min = features.min()
    if in_max is None:
        in_max = features.max()
    
    eps = 1e-8
    range_in = in_max - in_min + eps
    
    normalized = (features - in_min) / range_in
    scaled = normalized * (out_max - out_min) + out_min
    
    return scaled


def log_angle_statistics(
    angles: torch.Tensor,
    label: str = "rotation_angles",
) -> None:
    """
    Print debug statistics for rotation angles in a batch.
    
    Call this after normalization to verify the angle distribution
    is healthy (well-spread, no clipping, reasonable range).
    
    Args:
        angles: Tensor of rotation angles [batch, n_qubits]
        label: Descriptive label for the log line
    """
    with torch.no_grad():
        a = angles.detach()
        print(
            f"  [QDebug] {label}: "
            f"shape={list(a.shape)}, "
            f"mean={a.mean().item():.4f}, "
            f"std={a.std().item():.4f}, "
            f"min={a.min().item():.4f}, "
            f"max={a.max().item():.4f}"
        )


def angle_encoding(
    features: torch.Tensor,
    method: str = 'ry',
    scale: float = np.pi
) -> torch.Tensor:
    """
    Apply angle encoding transformation to features.
    
    Different encoding methods have different properties:
    - 'ry': Direct angle for RY gate, range [0, π]
    - 'arctan': Arctan transformation, bounded output
    - 'arcsin': Arcsin transformation, compressed extremes
    
    Args:
        features: Input features [batch, n_features]
        method: Encoding method ('ry', 'arctan', 'arcsin')
        scale: Scaling factor for the encoding
        
    Returns:
        Encoded features suitable for quantum gates
    """
    if method == 'ry':
        # Simple linear scaling to [0, scale]
        return normalize_features(features, 0, scale)
    
    elif method == 'arctan':
        # Arctan encoding: bounded, handles outliers well
        # Output range: (-π/2, π/2)
        return torch.arctan(features)
    
    elif method == 'arcsin':
        # Arcsin encoding: requires features in [-1, 1]
        # Clamp to valid range first
        clamped = torch.clamp(features, -1 + 1e-6, 1 - 1e-6)
        return torch.arcsin(clamped)
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def dense_angle_encoding(features: torch.Tensor) -> torch.Tensor:
    """
    Dense angle encoding for more expressive feature mapping.
    
    Uses multiple rotations per feature to increase expressivity.
    Each feature is encoded as: [cos(f), sin(f), f]
    
    This is useful when the quantum circuit is shallow and
    needs more information per qubit.
    
    Args:
        features: Input features [batch, n_features]
        
    Returns:
        Dense encoded features [batch, n_features * 3]
    """
    # Normalize to [0, 2π]
    normalized = normalize_features(features, 0, 2 * np.pi)
    
    # Create dense encoding
    cos_f = torch.cos(normalized)
    sin_f = torch.sin(normalized)
    
    # Stack along feature dimension
    dense = torch.cat([cos_f, sin_f, normalized], dim=-1)
    
    return dense


def get_quantum_device(
    config: Optional[QuantumConfig] = None,
    device_name: str = 'default.qubit',
    n_qubits: int = 10,
    shots: Optional[int] = None
):
    """
    Create a PennyLane quantum device.
    
    Args:
        config: Optional quantum configuration
        device_name: PennyLane device name
        n_qubits: Number of qubits
        shots: Number of measurement shots (None for analytic)
        
    Returns:
        PennyLane device instance
    """
    if config is not None:
        device_name = config.device
        n_qubits = config.n_qubits
        shots = config.shots
    
    return qml.device(device_name, wires=n_qubits, shots=shots)


def compute_circuit_resources(
    n_qubits: int,
    n_layers: int,
    entanglement: str = 'linear'
) -> dict:
    """
    Compute resource requirements for a quantum circuit.
    
    Useful for understanding circuit complexity and
    compatibility with hardware.
    
    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        entanglement: Entanglement pattern
        
    Returns:
        Dictionary with resource counts
    """
    # Single-qubit gates: RY for encoding + RY for each variational layer
    encoding_gates = n_qubits
    variational_ry = n_qubits * n_layers
    
    # CNOT gates per entanglement pattern
    if entanglement == 'linear':
        cnots_per_layer = n_qubits - 1
    elif entanglement == 'ring':
        cnots_per_layer = n_qubits
    elif entanglement == 'full':
        cnots_per_layer = n_qubits * (n_qubits - 1) // 2
    else:
        cnots_per_layer = 0
    
    total_cnots = cnots_per_layer * n_layers
    
    # Estimate circuit depth
    # Assuming single-qubit gates can be parallelized
    depth = 1  # Encoding layer
    for _ in range(n_layers):
        depth += 1  # Variational RY layer
        depth += cnots_per_layer  # CNOT layer (worst case)
    
    # Trainable parameters
    n_params = n_qubits * n_layers
    
    return {
        'n_qubits': n_qubits,
        'n_layers': n_layers,
        'single_qubit_gates': encoding_gates + variational_ry,
        'two_qubit_gates': total_cnots,
        'circuit_depth': depth,
        'trainable_parameters': n_params,
        'entanglement': entanglement,
    }


def print_circuit_info(config: QuantumConfig) -> None:
    """Print circuit resource information."""
    resources = compute_circuit_resources(
        config.n_qubits,
        config.n_layers,
        config.entanglement
    )
    
    print("\n" + "=" * 50)
    print("Quantum Circuit Resources")
    print("=" * 50)
    for key, value in resources.items():
        print(f"  {key}: {value}")
    print("=" * 50)


def validate_quantum_config(config: QuantumConfig) -> bool:
    """
    Validate quantum configuration for common issues.
    
    Args:
        config: Quantum configuration to validate
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check qubit count
    if config.n_qubits > 30:
        raise ValueError(
            f"n_qubits ({config.n_qubits}) is too large for classical simulation. "
            "Consider reducing to ≤20 for reasonable performance."
        )
    
    # Check layer count for barren plateau risk
    if config.n_layers > 10:
        print(
            f"Warning: n_layers ({config.n_layers}) is high. "
            "Consider reducing to avoid barren plateau issues."
        )
    
    # Validate entanglement pattern
    valid_patterns = ['linear', 'ring', 'full']
    if config.entanglement not in valid_patterns:
        raise ValueError(
            f"Invalid entanglement pattern: {config.entanglement}. "
            f"Must be one of {valid_patterns}"
        )
    
    # Check output dimension
    if config.n_outputs > config.n_qubits:
        raise ValueError(
            f"n_outputs ({config.n_outputs}) cannot exceed n_qubits ({config.n_qubits})"
        )
    
    return True


def quantum_state_tomography(
    circuit_fn,
    weights: torch.Tensor,
    n_qubits: int
) -> np.ndarray:
    """
    Perform simple state tomography on a quantum circuit.
    
    Returns the probability amplitudes of the quantum state
    after circuit execution. Useful for debugging and analysis.
    
    Note: This only works with simulators, not real hardware.
    
    Args:
        circuit_fn: QNode circuit function
        weights: Circuit parameters
        n_qubits: Number of qubits
        
    Returns:
        Complex amplitudes array [2^n_qubits]
    """
    # Create a special circuit that returns state vector
    dev = qml.device('default.qubit', wires=n_qubits)
    
    @qml.qnode(dev)
    def state_circuit(inputs, weights):
        # Execute the same gates as the original circuit
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        
        n_layers = weights.shape[0]
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        return qml.state()
    
    # Get state with zero input (for analysis)
    zero_input = np.zeros(n_qubits)
    state = state_circuit(zero_input, weights.detach().numpy())
    
    return np.array(state)


def measure_entanglement_entropy(
    state: np.ndarray,
    n_qubits: int,
    partition: int = None
) -> float:
    """
    Measure entanglement entropy of a quantum state.
    
    Computes the von Neumann entropy of the reduced density matrix
    for a bipartition of the system.
    
    Args:
        state: Quantum state vector [2^n_qubits]
        n_qubits: Number of qubits
        partition: Qubit to partition at (default: n_qubits // 2)
        
    Returns:
        Entanglement entropy (bits)
    """
    if partition is None:
        partition = n_qubits // 2
    
    # Reshape state into bipartite tensor
    dim_a = 2 ** partition
    dim_b = 2 ** (n_qubits - partition)
    psi = state.reshape(dim_a, dim_b)
    
    # Compute reduced density matrix
    rho_a = psi @ psi.conj().T
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(rho_a)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
    
    # Compute von Neumann entropy
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return float(entropy)
