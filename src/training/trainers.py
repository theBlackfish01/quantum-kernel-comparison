from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from src.models import BaseKernel

class SVCTrainer:
    """Trainer for Support Vector Classifier with custom kernels."""
    
    def __init__(self, kernel: BaseKernel):
        self.kernel = kernel
        self.model = SVC(kernel=self._kernel_wrapper)
        
    def _kernel_wrapper(self, X1, X2):
        """Wrapper to make the kernel class compatible with sklearn SVC."""
        # SVC passes numpy arrays
        return self.kernel.kernel_matrix(X1, X2)
        
    def train(self, X_train, y_train):
        """Fits the model to the training data."""
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data.
        
        Returns:
            dict: Dictionary containing accuracy and predictions.
        """
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return {
            "accuracy": accuracy,
            "predictions": predictions
        }
