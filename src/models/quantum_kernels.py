import pennylane as qml
from pennylane import numpy as np
from abc import ABC, abstractmethod

class BaseKernel(ABC):
    """Abstract base class for kernels."""
    @abstractmethod
    def evaluate(self, x1, x2):
        """Evaluate kernel function k(x1, x2)."""
        pass
    
    @abstractmethod
    def kernel_matrix(self, X1, X2):
        """Compute the kernel matrix between X1 and X2."""
        pass

class QuantumKernel(BaseKernel):
    """Quantum Kernel Based on Projected Quantum Kernel protocol."""
    
    def __init__(self, num_wires: int, num_layers: int, dev_name: str = "default.qubit", shots=None):
        self.num_wires = num_wires
        self.num_layers = num_layers
        # Support Qiskit device creation
        self.dev = qml.device(dev_name, wires=num_wires, shots=shots)
        
        # Initialize random parameters
        # Shape: (num_layers, 2, num_wires)
        # explicitly requires_grad=True is default for pennylane numpy if not specified, 
        # but good to be explicit for optimization.
        self.params = np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)
        
        # Define the QNode closure
        @qml.qnode(self.dev, interface="autograd") # Specify interface for gradient descent
        def _circuit(x1, x2, params):
            self._ansatz(x1, params, wires=self.dev.wires)
            qml.adjoint(self._ansatz)(x2, params, wires=self.dev.wires)
            return qml.probs(wires=self.dev.wires)
            
        self._circuit = _circuit

    def _layer(self, x, params, wires, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for j, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            qml.RY(params[0, j], wires=[wire])

        for i in range(len(wires)):
            qml.CRZ(params[1][i], wires=[wires[i], wires[(i + 1) % len(wires)]])

    def _ansatz(self, x, params, wires):
        """The embedding ansatz"""
        for j, layer_params in enumerate(params):
            self._layer(x, layer_params, wires, i0=j * len(wires))

    def evaluate(self, x1, x2):
        """Returns the probability of the all-zero state."""
        return self._circuit(x1, x2, self.params)[0]
        
    def kernel_matrix(self, X1, X2):
        """Computes the full kernel matrix."""
        # Wrap the evaluate method to be compatible with qml.kernels.kernel_matrix
        # qml.kernels.kernel_matrix expects a function func(x1, x2) -> float
        k_func = lambda x1, x2: self.evaluate(x1, x2)
        return qml.kernels.kernel_matrix(X1, X2, k_func)
