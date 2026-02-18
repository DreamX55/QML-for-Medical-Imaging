"""
Quantum Module for Hybrid Quantum-Classical ML.

Provides quantum circuit definitions, encoding strategies,
and PyTorch integration via PennyLane.
"""

from .circuits import create_quantum_circuit, ParameterizedQuantumCircuit
from .quantum_layer import QuantumLayer
from .utils import normalize_features, angle_encoding, get_quantum_device

__all__ = [
    "create_quantum_circuit",
    "ParameterizedQuantumCircuit",
    "QuantumLayer",
    "normalize_features",
    "angle_encoding",
    "get_quantum_device",
]
