import numpy as np
from typing import Callable, Tuple

class TestFunctions:
    """Collection of test functions for optimizer benchmarking"""
    
    @staticmethod
    def quadratic(x: np.ndarray, A: np.ndarray = None, b: np.ndarray = None) -> float:
        """Quadratic function: f(x) = ½xᵀAx + bᵀx"""
        if A is None:
            A = np.array([[2.0, 1.0], [1.0, 3.0]])
        if b is None:
            b = np.array([1.0, 2.0])
        return 0.5 * x.T @ A @ x + b.T @ x
    
    @staticmethod
    def quadratic_gradient(x: np.ndarray, A: np.ndarray = None, b: np.ndarray = None) -> np.ndarray:
        """Gradient of quadratic function: ∇f(x) = Ax + b"""
        if A is None:
            A = np.array([[2.0, 1.0], [1.0, 3.0]])
        if b is None:
            b = np.array([1.0, 2.0])
        return A @ x + b
    
    @staticmethod
    def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
        """
        Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        Global minimum at (a, a²)
        """
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    @staticmethod
    def rosenbrock_gradient(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
        """Gradient of Rosenbrock function"""
        dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
        dy = 2 * b * (x[1] - x[0]**2)
        return np.array([dx, dy])
    
    @staticmethod
    def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
        """
        Rastrigin function: f(x) = A*n + Σ[x_i² - A*cos(2πx_i)]
        Global minimum at (0, 0, ..., 0)
        """
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rastrigin_gradient(x: np.ndarray, A: float = 10.0) -> np.ndarray:
        """Gradient of Rastrigin function"""
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    
    @staticmethod
    def beale(x: np.ndarray) -> float:
        """
        Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
        Global minimum at (3, 0.5)
        """
        term1 = (1.5 - x[0] + x[0] * x[1]) ** 2
        term2 = (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        term3 = (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
        return term1 + term2 + term3
    
    @staticmethod
    def beale_gradient(x: np.ndarray) -> np.ndarray:
        """Gradient of Beale function"""
        term1_dx = 2 * (1.5 - x[0] + x[0] * x[1]) * (-1 + x[1])
        term1_dy = 2 * (1.5 - x[0] + x[0] * x[1]) * x[0]
        
        term2_dx = 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (-1 + x[1] ** 2)
        term2_dy = 2 * (2.25 - x[0] + x[0] * x[1] ** 2) * (2 * x[0] * x[1])
        
        term3_dx = 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (-1 + x[1] ** 3)
        term3_dy = 2 * (2.625 - x[0] + x[0] * x[1] ** 3) * (3 * x[0] * x[1] ** 2)
        
        return np.array([term1_dx + term2_dx + term3_dx, 
                        term1_dy + term2_dy + term3_dy])