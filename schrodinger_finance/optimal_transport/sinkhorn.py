"""
Sinkhorn Algorithm for Optimal Transport
=========================================

Implementation of Sinkhorn-Knopp algorithm for entropic regularized
optimal transport and Sinkhorn divergence for model calibration.
"""

import numpy as np
from scipy.special import logsumexp
from typing import Optional, Tuple, Callable
import warnings


class SinkhornSolver:
    """
    Sinkhorn algorithm for optimal transport with entropic regularization.
    
    Solves:
        min_P <C, P> + ε H(P)
        s.t. P 1 = a, P^T 1 = b
        
    where H(P) = Σ_ij P_ij log(P_ij) is the entropy.
    """
    
    def __init__(
        self,
        eps: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-9,
        device: str = 'cpu'
    ):
        """
        Initialize Sinkhorn solver.
        
        Parameters
        ----------
        eps : float
            Entropic regularization parameter (default: 0.01)
        max_iter : int
            Maximum iterations (default: 1000)
        tol : float
            Convergence tolerance (default: 1e-9)
        device : str
            Computation device ('cpu' or 'cuda')
        """
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        
    def solve(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        return_transport: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Solve OT problem using Sinkhorn algorithm.
        
        Parameters
        ----------
        a : array of shape (n,)
            Source distribution
        b : array of shape (m,)
            Target distribution
        C : array of shape (n, m)
            Cost matrix
        return_transport : bool
            If True, return transport plan
            
        Returns
        -------
        cost : float
            Optimal transport cost
        P : array, optional
            Optimal transport plan
        """
        n, m = C.shape
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        
        # Normalize
        a = a / (a.sum() + 1e-16)
        b = b / (b.sum() + 1e-16)
        
        # Gibbs kernel
        K = np.exp(-C / self.eps)
        
        # Initialize dual variables
        u = np.ones(n)
        v = np.ones(m)
        
        for iteration in range(self.max_iter):
            u_prev = u.copy()
            
            # Sinkhorn iterations
            v = b / (K.T @ u + 1e-16)
            u = a / (K @ v + 1e-16)
            
            # Check convergence
            if np.max(np.abs(u - u_prev)) < self.tol:
                break
        
        # Transport plan
        P = np.diag(u) @ K @ np.diag(v)
        
        # Transport cost
        cost = np.sum(P * C)
        
        if return_transport:
            return cost, P
        return cost, None
    
    def solve_log_stable(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Log-domain Sinkhorn for numerical stability.
        
        Better for small epsilon values.
        """
        n, m = C.shape
        a = a / (a.sum() + 1e-16)
        b = b / (b.sum() + 1e-16)
        
        # Log kernel
        log_K = -C / self.eps
        
        # Log dual variables
        log_u = np.zeros(n)
        log_v = np.zeros(m)
        
        for iteration in range(self.max_iter):
            log_u_prev = log_u.copy()
            
            # Log-sum-exp updates
            log_u = np.log(a + 1e-16) - logsumexp(log_K + log_v[np.newaxis, :], axis=1)
            log_v = np.log(b + 1e-16) - logsumexp(log_K.T + log_u[np.newaxis, :], axis=1)
            
            if np.max(np.abs(log_u - log_u_prev)) < self.tol:
                break
        
        # Transport plan
        log_P = log_u[:, np.newaxis] + log_K + log_v[np.newaxis, :]
        P = np.exp(log_P)
        
        cost = np.sum(P * C)
        
        return cost, P
    
    def sinkhorn_divergence(
        self,
        a: np.ndarray,
        b: np.ndarray,
        C: np.ndarray,
        C_aa: Optional[np.ndarray] = None,
        C_bb: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Sinkhorn divergence.
        
        S(a, b) = OT_ε(a, b) - 0.5 * OT_ε(a, a) - 0.5 * OT_ε(b, b)
        
        Parameters
        ----------
        a, b : arrays
            Distributions to compare
        C : array
            Cost matrix between a and b
        C_aa, C_bb : arrays, optional
            Self-cost matrices (computed from C if not provided)
            
        Returns
        -------
        divergence : float
            Sinkhorn divergence value
        """
        # OT(a, b)
        cost_ab, _ = self.solve(a, b, C, return_transport=False)
        
        # Self-transport terms
        if C_aa is None:
            n = len(a)
            C_aa = self._compute_self_cost(a)
        if C_bb is None:
            m = len(b)
            C_bb = self._compute_self_cost(b)
        
        cost_aa, _ = self.solve(a, a, C_aa, return_transport=False)
        cost_bb, _ = self.solve(b, b, C_bb, return_transport=False)
        
        return cost_ab - 0.5 * cost_aa - 0.5 * cost_bb
    
    def _compute_self_cost(self, x: np.ndarray) -> np.ndarray:
        """Compute self-cost matrix for uniform distribution."""
        n = len(x)
        # Simple Euclidean distance cost
        indices = np.arange(n)
        C = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :]).astype(float)
        return C
    
    def wasserstein_distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        p: int = 2
    ) -> float:
        """
        Compute p-Wasserstein distance approximation via Sinkhorn.
        
        Parameters
        ----------
        a, b : arrays
            Distributions (histograms)
        p : int
            Wasserstein order (default: 2)
            
        Returns
        -------
        distance : float
            Approximate Wasserstein distance
        """
        n, m = len(a), len(b)
        
        # Create cost matrix (assuming 1D distributions)
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, m)
        C = np.abs(x[:, np.newaxis] - y[np.newaxis, :]) ** p
        
        cost, _ = self.solve(a, b, C, return_transport=False)
        
        return cost ** (1.0 / p)


class SinkhornCalibrator:
    """
    Model calibration using Sinkhorn divergence.
    
    Calibrates model parameters by minimizing the Sinkhorn divergence
    between model-implied and market-observed distributions.
    """
    
    def __init__(
        self,
        model,
        eps: float = 0.05,
        lr: float = 0.01,
        max_iter: int = 100
    ):
        """
        Initialize calibrator.
        
        Parameters
        ----------
        model : object
            Model with simulate() method and params attribute
        eps : float
            Sinkhorn regularization
        lr : float
            Learning rate
        max_iter : int
            Calibration iterations
        """
        self.model = model
        self.eps = eps
        self.lr = lr
        self.max_iter = max_iter
        self.sinkhorn = SinkhornSolver(eps=eps)
        
    def calibrate(
        self,
        market_data: np.ndarray,
        n_samples: int = 1000,
        verbose: bool = True
    ) -> dict:
        """
        Calibrate model parameters.
        
        Parameters
        ----------
        market_data : array
            Observed market prices/returns
        n_samples : int
            Number of model samples per iteration
        verbose : bool
            Print progress
            
        Returns
        -------
        result : dict
            Calibrated parameters and history
        """
        # Create histogram from market data
        n_bins = 50
        market_hist, edges = np.histogram(market_data, bins=n_bins, density=True)
        market_hist = market_hist / market_hist.sum()
        
        history = {'losses': [], 'params': []}
        
        for iteration in range(self.max_iter):
            # Simulate from model
            model_samples = self.model.simulate(n_samples)
            model_hist, _ = np.histogram(model_samples, bins=edges, density=True)
            model_hist = model_hist / (model_hist.sum() + 1e-16)
            
            # Compute Sinkhorn divergence
            n = len(market_hist)
            x = np.linspace(0, 1, n)
            C = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
            
            loss = self.sinkhorn.sinkhorn_divergence(market_hist, model_hist, C)
            
            # Compute gradients numerically
            grads = self._compute_gradients(model_samples, market_hist, edges)
            
            # Update parameters
            self._update_params(grads)
            
            history['losses'].append(loss)
            history['params'].append(self._get_params())
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Sinkhorn Loss = {loss:.6f}")
        
        return {
            'calibrated_params': self._get_params(),
            'history': history
        }
    
    def _compute_gradients(
        self,
        samples: np.ndarray,
        target_hist: np.ndarray,
        edges: np.ndarray
    ) -> dict:
        """Compute numerical gradients of parameters."""
        grads = {}
        eps = 1e-4
        
        for name, param in self.model.params.items():
            # Perturb parameter
            original = param.copy() if hasattr(param, 'copy') else param
            
            # Forward difference
            self.model.params[name] = original + eps
            samples_plus = self.model.simulate(len(samples))
            hist_plus, _ = np.histogram(samples_plus, bins=edges, density=True)
            hist_plus = hist_plus / (hist_plus.sum() + 1e-16)
            
            # Compute loss
            n = len(target_hist)
            x = np.linspace(0, 1, n)
            C = (x[:, np.newaxis] - x[np.newaxis, :]) ** 2
            loss_plus = self.sinkhorn.sinkhorn_divergence(target_hist, hist_plus, C)
            
            # Restore
            self.model.params[name] = original
            
            # Backward difference
            self.model.params[name] = original - eps
            samples_minus = self.model.simulate(len(samples))
            hist_minus, _ = np.histogram(samples_minus, bins=edges, density=True)
            hist_minus = hist_minus / (hist_minus.sum() + 1e-16)
            loss_minus = self.sinkhorn.sinkhorn_divergence(target_hist, hist_minus, C)
            
            # Gradient
            grads[name] = (loss_plus - loss_minus) / (2 * eps)
            self.model.params[name] = original
        
        return grads
    
    def _update_params(self, grads: dict):
        """Update parameters using gradients."""
        for name, grad in grads.items():
            if name in self.model.params:
                self.model.params[name] -= self.lr * grad
                
                # Apply constraints
                if name in ['kappa', 'theta', 'sigma_v', 'v0']:
                    self.model.params[name] = max(self.model.params[name], 1e-4)
                elif name == 'rho':
                    self.model.params[name] = np.clip(self.model.params[name], -0.99, 0.99)
    
    def _get_params(self) -> dict:
        """Get current parameters."""
        return dict(self.model.params)
