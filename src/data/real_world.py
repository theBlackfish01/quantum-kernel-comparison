import numpy as np
from sklearn.datasets import make_moons, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_moons_data(n_samples=200, noise=0.1):
    """
    Generate Two Moons dataset.
    Returns: X, y (already in canonical -1, 1 labels if needed, though sklearn uses 0, 1)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    # Convert labels 0, 1 to -1, 1 for consistency with hinge loss / typical SVM
    y = 2 * y - 1
    return X, y.astype(int)

def load_digits_data(n_samples=None, n_features=4, classes=(0, 1)):
    """
    Load MNIST Digits, filter to two classes, and reduce dimensions via PCA.
    Args:
        n_samples: Total samples to load (if None, load all)
        n_features: target dimensionality (number of qubits)
        classes: tuple of two integers to classify
    """
    digits = load_digits()
    
    # Filter classes
    mask = np.isin(digits.target, classes)
    X = digits.data[mask]
    y = digits.target[mask]
    
    if n_samples:
        X = X[:n_samples]
        y = y[:n_samples]
        
    # Map labels to -1, 1
    # Assuming classes are e.g. (0, 1), map min->-1, max->1
    y_mapped = np.where(y == min(classes), -1, 1)
    
    # Standardize first (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA Reduction
    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y_mapped
