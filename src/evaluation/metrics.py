import pennylane as qml

def calculate_kta(kernel, X, Y):
    """Calculate Kernel Target Alignment.
    
    Args:
        kernel: QuantumKernel instance
        X: Data points
        Y: Labels
        
    Returns:
        float: KTA score
    """
    # qml.kernels.target_alignment expects a kernel function k(x1, x2)
    k_func = lambda x1, x2: kernel.evaluate(x1, x2)
    return qml.kernels.target_alignment(X, Y, k_func, assume_normalized_kernel=True)
