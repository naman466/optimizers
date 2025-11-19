import numpy as np

class Parameter:
    """
    A simple parameter class that holds values and gradients
    """
    def __init__(self, value: np.ndarray):
        self.value = np.array(value, dtype=np.float64)
        self.grad = None
        self.shape = self.value.shape
        
    def zero_grad(self):
        """Reset gradient to zero"""
        self.grad = None
        
    def __repr__(self):
        return f"Parameter(shape={self.shape}, value_norm={np.linalg.norm(self.value):.4f})"