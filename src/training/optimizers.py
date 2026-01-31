import pennylane as qml
from pennylane import numpy as np
from src.models import QuantumKernel

class KernelOptimizer:
    """Optimizes the Quantum Kernel parameters to maximize Target Alignment."""
    
    def __init__(self, kernel: QuantumKernel, learning_rate: float = 0.2, steps: int = 20):
        self.kernel = kernel
        self.lr = learning_rate
        self.steps = steps
        self.opt = qml.GradientDescentOptimizer(learning_rate)
        
    def target_alignment_loss(self, params, X, Y):
        """
        Loss function: Negative Kernel-Target Alignment.
        We want to MAXIMIZE Alignment, so we MINIMIZE negative Alignment.
        """
        # Update kernel parameters temporarily for this evaluation
        # Note: In PennyLane, we pass params to the qnode. 
        # But our kernel.evaluate uses self.params. 
        # We need to ensure the qnode sees the 'params' argument being differentiated.
        
        # We define a helper KTA function that depends explicitly on 'params'
        def k_func(x1, x2):
            return self.kernel._circuit(x1, x2, params)[0]
            
        kta = qml.kernels.target_alignment(X, Y, k_func, assume_normalized_kernel=True)
        return -kta

    def optimize(self, X_train, y_train):
        """Runs the optimization loop."""
        params = self.kernel.params
        
        print(f"Starting Kernel Optimization (Steps: {self.steps}, LR: {self.lr})")
        
        for i in range(self.steps):
            # Gradient Descent Step
            # The lambda allows us to pass X_train, y_train as fixed args
            cost_fn = lambda p: self.target_alignment_loss(p, X_train, y_train)
            
            params, loss = self.opt.step_and_cost(cost_fn, params)
            
            if (i + 1) % 5 == 0:
                print(f"Step {i+1} - KTA Loss: {loss:.4f} (Alignment: {-loss:.4f})")
                
        # Update the kernel with optimized parameters
        self.kernel.params = params
        print("Kernel Optimization Complete.")
        return params
