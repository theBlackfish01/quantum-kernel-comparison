from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data.synthetic import make_double_cake_data
from src.data.real_world import load_moons_data, load_digits_data

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    type: str = "double_cake"
    num_sectors: int = 3
    test_size: float = 0.3
    random_state: int = 42
    use_scaler: bool = False
    n_samples: int = 100 # Default sample size

class DataManager:
    """Manages dataset loading, preprocessing, and splitting."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.scaler = StandardScaler() if config.use_scaler else None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads data, splits it, and optionally scales it.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.config.type == "double_cake":
            X, y = make_double_cake_data(self.config.num_sectors)
        elif self.config.type == "moons":
            # Using random_state from config for reproducibility
            X, y = load_moons_data(n_samples=self.config.n_samples) 
        elif self.config.type == "digits":
             # Use config to determine PCA components if needed, defaulting to 4
             n_features = 4  
             X, y = load_digits_data(n_samples=self.config.n_samples, n_features=n_features)
        else:
            raise ValueError(f"Unknown data type: {self.config.type}")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test
