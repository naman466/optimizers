import numpy as np
from typing import List, Optional
from .parameter import Parameter

class Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer from scratch
    Implementation of: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
    """
    
    def __init__(self, parameters: List[Parameter], lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        """
        Parameters:
        -----------
        parameters : list of Parameter objects
            Model parameters to optimize
        lr : float
            Learning rate
        betas : tuple (beta1, beta2)
            Coefficients for computing running averages
        eps : float
            Term to improve numerical stability
        weight_decay : float
            L2 regularization (weight decay)
        amsgrad : bool
            Whether to use AMSGrad variant
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        
        # State initialization
        self.t = 0
        
        # Initialize moment arrays with same shape as parameters
        self.m = [np.zeros_like(p.value) for p in parameters]
        self.v = [np.zeros_like(p.value) for p in parameters]
        
        if amsgrad:
            self.v_hat = [np.zeros_like(p.value) for p in parameters]
        
        # Tracking for analysis
        self.history = {
            'loss': [],
            'grad_norms': [],
            'update_norms': [],
            'effective_lrs': [],
            'm_norms': [],
            'v_norms': [],
            'positions': []
        }
    
    def zero_grad(self):
        """Clear gradients from all parameters"""
        for param in self.parameters:
            param.zero_grad()
    
    def step(self):
        """
        Perform a single optimization step using stored gradients
        """
        self.t += 1
        
        # Bias correction terms
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        grad_norms = []
        update_norms = []
        effective_lrs = []
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            grad = param.grad
            
            # Apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.value
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate  
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected estimates
            m_hat = self.m[i] / bias_correction1
            
            if self.amsgrad:
                # Maintains the maximum of all v until now
                self.v_hat[i] = np.maximum(self.v_hat[i], self.v[i])
                v_hat = self.v_hat[i] / bias_correction2
            else:
                v_hat = self.v[i] / bias_correction2
            
            # Update parameters with adaptive learning rate
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.value -= update
            
            # Track statistics for analysis
            grad_norms.append(np.linalg.norm(grad))
            update_norms.append(np.linalg.norm(update))
            
            # Effective learning rate (average of |update| / |param|)
            with np.errstate(divide='ignore', invalid='ignore'):
                effective_lr = np.nanmean(np.abs(update) / (np.abs(param.value) + 1e-8))
                effective_lrs.append(effective_lr if not np.isnan(effective_lr) else 0)
        
        # Store history
        if grad_norms:
            self.history['grad_norms'].append(np.mean(grad_norms))
            self.history['update_norms'].append(np.mean(update_norms))
            self.history['effective_lrs'].append(np.mean(effective_lrs))
            self.history['m_norms'].append(np.mean([np.linalg.norm(m) for m in self.m]))
            self.history['v_norms'].append(np.mean([np.linalg.norm(v) for v in self.v]))
            
            # Store current parameter values for trajectory plotting
            if len(self.parameters) == 1:
                self.history['positions'].append(self.parameters[0].value.copy())
    
    def get_effective_learning_rates(self):
        """Compute per-parameter effective learning rates"""
        if self.t == 0:
            return [np.zeros_like(p.value) for p in self.parameters]
            
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t
        
        effective_lrs = []
        for i in range(len(self.parameters)):
            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2
            effective_lr = self.lr / (np.sqrt(v_hat) + self.eps)
            effective_lrs.append(effective_lr)
        
        return effective_lrs