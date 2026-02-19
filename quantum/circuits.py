"""
Parameterized Quantum Circuit (PQC) Definitions.

This module provides quantum circuit architectures for hybrid
quantum-classical machine learning. Uses PennyLane for
quantum simulation with PyTorch integration.

Design Decisions:
- 10-qubit architecture matching CNN feature compression
- RY gate encoding (angle encoding) for classical features
- Shallow circuit depth to avoid barren plateaus
- Linear entanglement pattern (CNOT chain) for efficiency
- Expectation values as outputs for gradient computation

Research Notes:
- Shallow circuits (2-3 layers) work better for NISQ and simulators
- Linear entanglement is hardware-efficient and avoids overhead
- RY encoding is a common choice for real-valued features
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Callable, Optional, Tuple, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import QuantumConfig


class ParameterizedQuantumCircuit:
    """
    Expressive Parameterized Quantum Circuit for hybrid ML.
    
    Architecture (per forward pass):
    1. Feature encoding layer: RY(feature_i) on each qubit
    2. Variational layers (×n_layers), each containing:
       - Trainable RY(θ) + RZ(φ) on every qubit
       - Ring entanglement (CNOT chain + last→first)
    3. Final trainable rotation layer: RY(θ) + RZ(φ) on every qubit
    4. Measurement: ⟨Z_i⟩ expectation values
    
    WHY THIS DESIGN INCREASES EXPRESSIVITY:
    
      The old circuit used only RY rotations. RY rotates around the Y-axis
      of the Bloch sphere, giving access to only ONE degree of freedom (the
      polar angle θ). This means the circuit can only explore a 1D subspace
      of each qubit's state space per layer.
    
      Adding RZ rotations gives access to the AZIMUTHAL angle φ, enabling
      the circuit to reach ANY point on the Bloch sphere — the full SU(2)
      per qubit. Concretely:
    
        RY-only:     |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩   (real amplitudes only)
        RY + RZ:     |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩  (complex phases)
    
      The complex phases are critical because:
        1. They interact non-trivially with CNOT gates, creating richer
           entanglement patterns that encode more complex decision boundaries.
        2. Without RZ, all amplitudes stay real, and CNOT on real states
           produces only a limited class of entangled states.
        3. The final rotation layer (after last entanglement) gives each
           qubit one last chance to align its measurement basis with the
           optimal direction, increasing classification sensitivity.
    
      Ring entanglement (vs linear) adds the last→first CNOT, breaking
      the boundary effect where qubit 0 and qubit N-1 have fewer
      entangling connections, making information flow more symmetric.
    
    Trainable parameters:
      Old:  n_qubits × n_layers                           (e.g., 10 × 2 = 20)
      New:  n_qubits × (n_layers + 1) × 2                 (e.g., 10 × 3 × 2 = 60)
      
      3× more parameters in the same circuit depth — increased capacity
      without deeper circuits that risk barren plateaus.
    
    Attributes:
        n_qubits: Number of qubits (should match input features)
        n_layers: Number of variational layers
        entanglement: Entanglement pattern ('linear', 'ring', 'full')
        dev: PennyLane quantum device
        circuit: Compiled quantum circuit function
    """
    
    def __init__(self, config: QuantumConfig):
        """
        Initialize the PQC.
        
        Args:
            config: Quantum configuration object
        """
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        self.entanglement = config.entanglement
        self.n_outputs = config.n_outputs
        self.encoding_range = config.encoding_range
        
        # Create quantum device
        self.dev = qml.device(
            config.device,
            wires=self.n_qubits,
            shots=config.shots  # None for analytic simulation
        )
        
        # Trainable parameters:
        #   - n_layers variational layers: RY + RZ per qubit → 2 params each
        #   - 1 final rotation layer:      RY + RZ per qubit → 2 params each
        # Weight shape: [n_layers + 1, n_qubits, 2]
        #   axis 0: layer index (0..n_layers-1 = variational, n_layers = final)
        #   axis 1: qubit index
        #   axis 2: 0 = RY angle, 1 = RZ angle
        self.n_params = self.n_qubits * (self.n_layers + 1) * 2
        
        # Build the circuit
        self.circuit = self._build_circuit()
        
        print(f"Initialized PQC with {self.n_qubits} qubits, {self.n_layers} layers")
        print(f"  Rotations per layer: RY + RZ (2 params per qubit)")
        print(f"  Final rotation layer: YES (RY + RZ before measurement)")
        print(f"  Total trainable parameters: {self.n_params}")
        print(f"  Entanglement: {self.entanglement}")
        print(f"  Device: {config.device}")
    
    def _build_circuit(self) -> qml.QNode:
        """
        Build the quantum circuit as a QNode.
        
        Returns:
            PennyLane QNode representing the circuit
        """
        # lightning.qubit uses 'adjoint', default.qubit uses 'backprop'
        diff_method = 'adjoint' if 'lightning' in self.config.device else 'backprop'
        @qml.qnode(self.dev, interface='torch', diff_method=diff_method)
        def circuit(inputs, weights):
            """
            Quantum circuit with feature encoding and expressive variational layers.
            
            Args:
                inputs: Classical features to encode [n_qubits]
                weights: Trainable parameters [n_layers + 1, n_qubits, 2]
                    weights[layer, qubit, 0] = RY angle
                    weights[layer, qubit, 1] = RZ angle
                    weights[n_layers] = final rotation layer
                
            Returns:
                Expectation values of Pauli-Z on measured qubits
            """
            # ── Step 1: Feature encoding layer ──
            # Each classical feature is encoded as an RY rotation angle.
            # After this, qubit i is in state: cos(f_i/2)|0⟩ + sin(f_i/2)|1⟩
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # ── Step 2: Variational layers ──
            # Each layer applies per-qubit RY+RZ rotations followed by entanglement.
            # RY rotates around Y (polar angle), RZ rotates around Z (phase).
            # Together they can reach any point on the Bloch sphere.
            for layer in range(self.n_layers):
                # Trainable rotations: RY(θ) + RZ(φ) per qubit
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement layer
                self._apply_entanglement()
            
            # ── Step 3: Final trainable rotation before measurement ──
            # This lets each qubit rotate into the optimal measurement basis
            # AFTER the last entanglement layer. Without this, the measurement
            # basis is fixed and may not align with the learned representation.
            for i in range(self.n_qubits):
                qml.RY(weights[self.n_layers, i, 0], wires=i)
                qml.RZ(weights[self.n_layers, i, 1], wires=i)
            
            # ── Step 4: Measurement ──
            # Expectation values of Pauli-Z: ∈ [-1, 1]
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_outputs)]
        
        return circuit
    
    def _apply_entanglement(self) -> None:
        """Apply entanglement pattern based on configuration."""
        if self.entanglement == 'linear':
            # Linear entanglement: CNOT chain
            # Q0 -> Q1 -> Q2 -> ... -> Qn-1
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        elif self.entanglement == 'ring':
            # Ring entanglement: Linear + last-to-first connection
            # Breaks boundary asymmetry: qubit 0 and N-1 now have
            # equal entangling connectivity (2 CNOTs each)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Close the ring
            qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        elif self.entanglement == 'full':
            # Full entanglement: All-to-all (expensive!)
            # Not recommended for large circuits
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CNOT(wires=[i, j])
        
        else:
            raise ValueError(f"Unknown entanglement pattern: {self.entanglement}")
    
    def __call__(self, inputs, weights):
        """Execute the circuit."""
        return self.circuit(inputs, weights)
    
    def get_circuit_depth(self) -> int:
        """Estimate the circuit depth."""
        # Encoding layer: 1
        # Each variational layer: 2 (RY + RZ) + entanglement depth
        # Final rotation: 2 (RY + RZ)
        if self.entanglement == 'linear':
            entangle_depth = self.n_qubits - 1
        elif self.entanglement == 'ring':
            entangle_depth = self.n_qubits
        else:
            entangle_depth = self.n_qubits * (self.n_qubits - 1) // 2
        
        return 1 + self.n_layers * (2 + entangle_depth) + 2
    
    def draw(self) -> str:
        """
        Draw the circuit for visualization.
        
        Returns:
            String representation of the circuit
        """
        # Create dummy inputs for drawing
        dummy_inputs = pnp.zeros(self.n_qubits)
        dummy_weights = pnp.zeros((self.n_layers + 1, self.n_qubits, 2))
        
        drawer = qml.draw(self.circuit)
        return drawer(dummy_inputs, dummy_weights)


def create_quantum_circuit(config: QuantumConfig) -> ParameterizedQuantumCircuit:
    """
    Factory function to create a quantum circuit.
    
    Args:
        config: Quantum configuration
        
    Returns:
        ParameterizedQuantumCircuit instance
    """
    return ParameterizedQuantumCircuit(config)


class AlternativeEncodingCircuit:
    """
    Alternative quantum circuit with amplitude encoding.
    
    This is an alternative to angle encoding that encodes
    features into the amplitudes of the quantum state.
    
    Note: Amplitude encoding is more expressive but requires
    state preparation circuits which can be deep.
    For simulator-friendly experiments, angle encoding is preferred.
    """
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        # For amplitude encoding, we need 2^n_qubits >= n_features
        self.n_amplitudes = 2 ** self.n_qubits
        
        self.dev = qml.device(
            config.device,
            wires=self.n_qubits,
            shots=config.shots
        )
        
        self.circuit = self._build_circuit()
    
    def _build_circuit(self) -> qml.QNode:
        """Build amplitude encoding circuit."""
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # Pad inputs to match amplitude space
            padded = np.zeros(self.n_amplitudes)
            padded[:len(inputs)] = inputs
            
            # Normalize for valid quantum state
            norm = np.linalg.norm(padded)
            if norm > 0:
                padded = padded / norm
            else:
                padded[0] = 1.0
            
            # Amplitude encoding
            qml.AmplitudeEmbedding(padded, wires=range(self.n_qubits), normalize=True)
            
            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                    qml.RZ(weights[layer, i], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit


class DataReuploadingCircuit:
    """
    Data re-uploading quantum circuit.
    
    This architecture re-encodes data in each layer, providing
    more expressive power at the cost of deeper circuits.
    
    Research Note: Data re-uploading can improve expressivity
    and is proven to be a universal function approximator.
    """
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        self.dev = qml.device(
            config.device,
            wires=self.n_qubits,
            shots=config.shots
        )
        
        # Parameters per layer: RY + RZ for each qubit
        self.n_params = self.n_qubits * self.n_layers * 2
        
        self.circuit = self._build_circuit()
    
    def _build_circuit(self) -> qml.QNode:
        """Build data re-uploading circuit."""
        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            for layer in range(self.n_layers):
                # Re-upload data
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
                
                # Variational rotations
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_outputs)]
        
        return circuit
