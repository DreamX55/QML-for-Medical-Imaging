"""
Quantum Layer for PyTorch Integration.

This module provides a PyTorch-compatible quantum layer that
wraps PennyLane circuits for use in hybrid neural networks.

Design Decisions:
- Inherits from torch.nn.Module for seamless integration
- Handles batch processing (PennyLane processes samples individually)
- Manages trainable quantum parameters as PyTorch Parameters
- Supports gradient computation via PennyLane's autodiff
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

import pennylane as qml
from pennylane import numpy as pnp

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import QuantumConfig
from .circuits import ParameterizedQuantumCircuit
from .utils import normalize_features, log_angle_statistics


class QuantumLayer(nn.Module):
    """
    PyTorch-compatible Quantum Layer.
    
    Wraps a parameterized quantum circuit for use in hybrid
    neural networks. Handles batch processing and gradient
    computation automatically.
    
    The layer:
    1. Normalizes input features to the encoding range [0, π]
       using sigmoid-based scaling (input-independent, stable)
    2. Executes the quantum circuit on each sample
    3. Returns expectation values as output features
    
    Attributes:
        config: Quantum configuration
        pqc: Parameterized quantum circuit
        weights: Trainable quantum parameters
        n_qubits: Number of qubits (input dimension)
        n_outputs: Number of measurement outputs
    """
    
    def __init__(self, config: QuantumConfig):
        """
        Initialize the quantum layer.
        
        Args:
            config: Quantum configuration object
        """
        super().__init__()
        
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.n_outputs = config.n_outputs
        self.encoding_range = config.encoding_range
        
        # Create the quantum circuit
        self.pqc = ParameterizedQuantumCircuit(config)
        
        # Initialize trainable weights
        # Shape: [n_layers + 1, n_qubits, 2]
        #   axis 0: layer index (0..n_layers-1 = variational, n_layers = final rotation)
        #   axis 1: qubit index
        #   axis 2: 0 = RY angle (polar), 1 = RZ angle (azimuthal)
        # Initialized uniformly in [0, 2π] for symmetry breaking
        weight_init = np.random.uniform(
            low=0,
            high=2 * np.pi,
            size=(config.n_layers + 1, config.n_qubits, 2)
        )
        self.weights = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
        
        # Debug: log angle statistics for the first N forward passes
        self._debug_counter = 0
        self._debug_max = 3  # Print stats for first 3 batches only
        
        print(f"QuantumLayer initialized with {self.weights.numel()} trainable parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum layer.
        
        Args:
            x: Input tensor [batch_size, n_qubits]
               Features should be in a reasonable range for normalization
            
        Returns:
            Output tensor [batch_size, n_outputs]
            Contains expectation values in range [-1, 1]
        """
        batch_size = x.shape[0]
        original_device = x.device
        
        # Move to CPU for PennyLane (doesn't support MPS/CUDA directly)
        x_cpu = x.detach().cpu()
        weights_cpu = self.weights.cpu()
        
        # Normalize features to encoding range [0, π]
        # Uses sigmoid-based scaling: stable and input-independent.
        # The same feature value ALWAYS maps to the same rotation angle,
        # regardless of batch composition (unlike min/max normalization).
        x_normalized = normalize_features(
            x_cpu, 
            out_min=self.encoding_range[0],
            out_max=self.encoding_range[1]
        )
        
        # Debug: print angle statistics for the first few batches
        if self._debug_counter < self._debug_max:
            log_angle_statistics(x_cpu, label="pre_norm_features")
            log_angle_statistics(x_normalized, label="rotation_angles")
            self._debug_counter += 1
        
        # Execute quantum circuit with parameter broadcasting
        # PennyLane supports broadcasting when using diff_method='backprop':
        # passing [batch_size, n_qubits] inputs processes all samples in one
        # vectorized call instead of a Python for-loop over the batch.
        try:
            result = self.pqc(x_normalized, weights_cpu)
            if isinstance(result, (list, tuple)):
                # result is list of tensors, each [batch_size] → stack to [n_outputs, batch_size]
                output = torch.stack(result, dim=-1)  # [batch_size, n_outputs]
                if output.dim() == 1:
                    output = output.unsqueeze(0)
            else:
                output = result
        except Exception:
            # Fallback: sequential processing for backends without broadcasting
            outputs = []
            for i in range(batch_size):
                sample = x_normalized[i]
                result = self.pqc(sample, weights_cpu)
                if isinstance(result, list):
                    result = torch.stack(result)
                outputs.append(result)
            output = torch.stack(outputs, dim=0)
        
        # Convert to float32 and move back to original device
        # (MPS doesn't support float64, PennyLane may return float64)
        output = output.float().to(original_device)
        
        return output
    
    def get_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantum-enhanced features without gradient tracking.
        
        Useful for analysis and XAI purposes.
        
        Args:
            x: Input tensor [batch_size, n_qubits]
            
        Returns:
            Quantum features [batch_size, n_outputs]
        """
        with torch.no_grad():
            return self.forward(x)
    
    def get_circuit_visualization(self) -> str:
        """Get string visualization of the quantum circuit."""
        return self.pqc.draw()
    
    def get_num_parameters(self) -> int:
        """Get number of trainable quantum parameters."""
        return self.weights.numel()


class BatchQuantumLayer(nn.Module):
    """
    Optimized Quantum Layer with parallel batch processing.
    
    Uses parameter broadcasting for more efficient batch processing.
    This implementation creates multiple circuit evaluations
    that can potentially be parallelized.
    
    Note: This is an optimization for simulation. Real quantum
    hardware would still require sequential execution.
    """
    
    def __init__(self, config: QuantumConfig):
        super().__init__()
        
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.n_outputs = config.n_outputs
        
        # Create device
        self.dev = qml.device(
            config.device,
            wires=self.n_qubits,
            shots=config.shots
        )
        
        # Initialize weights
        weight_init = np.random.uniform(
            low=0,
            high=2 * np.pi,
            size=(config.n_layers, config.n_qubits)
        )
        self.weights = nn.Parameter(torch.tensor(weight_init, dtype=torch.float32))
        
        # Build batch-aware circuit
        self._build_circuit()
    
    def _build_circuit(self):
        """Build the quantum circuit with batch support."""
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # Feature encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                
                # Linear entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_outputs)]
        
        self.circuit = circuit
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with batch processing."""
        batch_size = x.shape[0]
        
        # Normalize
        x_norm = normalize_features(x, 0, np.pi)
        
        # Process batch
        outputs = []
        for i in range(batch_size):
            result = self.circuit(x_norm[i], self.weights)
            if isinstance(result, list):
                result = torch.stack(result)
            outputs.append(result)
        
        return torch.stack(outputs, dim=0)


class HybridQuantumDense(nn.Module):
    """
    Quantum layer with classical pre/post-processing.
    
    Combines:
    1. Classical linear transformation (pre-processing)
    2. Quantum circuit evaluation
    3. Classical linear transformation (post-processing)
    
    This architecture allows the model to learn optimal
    input encoding and output interpretation.
    """
    
    def __init__(
        self,
        config: QuantumConfig,
        in_features: int,
        out_features: int,
        use_pre_linear: bool = True,
        use_post_linear: bool = True
    ):
        """
        Initialize hybrid quantum-dense layer.
        
        Args:
            config: Quantum configuration
            in_features: Input feature dimension
            out_features: Output feature dimension
            use_pre_linear: Whether to use pre-quantum linear layer
            use_post_linear: Whether to use post-quantum linear layer
        """
        super().__init__()
        
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        
        # Pre-processing: map input to qubit count
        if use_pre_linear and in_features != config.n_qubits:
            self.pre_linear = nn.Linear(in_features, config.n_qubits)
        else:
            self.pre_linear = None
        
        # Quantum layer
        self.quantum = QuantumLayer(config)
        
        # Post-processing: map quantum output to desired dimension
        if use_post_linear and config.n_outputs != out_features:
            self.post_linear = nn.Linear(config.n_outputs, out_features)
        else:
            self.post_linear = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid layer."""
        # Pre-processing
        if self.pre_linear is not None:
            x = self.pre_linear(x)
            x = torch.tanh(x)  # Bound inputs before quantum
        
        # Quantum processing
        x = self.quantum(x)
        
        # Post-processing
        if self.post_linear is not None:
            x = self.post_linear(x)
        
        return x
