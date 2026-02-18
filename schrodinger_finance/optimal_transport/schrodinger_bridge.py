"""
Schrödinger Bridge Implementation
=================================

Implementation of the Schrödinger bridge problem for finding
the most likely stochastic process connecting two distributions.

Mathematical Foundation:
- Find measure P on path space minimizing KL(P||R)
- Subject to P₀ = μ₀, P_T = μ₁
- R is reference measure (usually Wiener measure)

Connection to Optimal Transport:
- Entropic regularization of OT
- Sinkhorn iterations provide solution
"""

import numpy as np
from typing import Optional, Tuple, Callable, Dict, Any
from scipy.special import logsumexp
from scipy.stats import norm
import warnings

from .sinkhorn import SinkhornSolver


class SchrodingerBridge:
    """
    Schrödinger Bridge solver for distribution matching.
    
    Finds optimal transport between distributions with
    entropy regularization (Brownian reference measure).
    """
    
    def __init__(
        self,
        dim: int,
        sigma: float = 1.0,
        n_steps: int = 100,
        eps: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        """
        Initialize Schrödinger bridge solver.
        
        Parameters
        ----------
        dim : int
            Dimension of state space
        sigma : float
            Diffusion coefficient
        n_steps : int
            Number of time discretization steps
        eps : float
            Entropic regularization
        max_iter : int
            Maximum Sinkhorn iterations
        tol : float
            Convergence tolerance
        """
        self.dim = dim
        self.sigma = sigma
        self.n_steps = n_steps
        self.eps = eps
        self.max_iter = max_iter
        self.tol = tol
        
        self.sinkhorn = SinkhornSolver(eps=eps, max_iter=max_iter, tol=tol)
        
    def compute_transition_kernel(
        self,
        x_grid: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute Brownian transition kernel.
        
        K(x, y) = N(y; x, σ²dt)
        
        Parameters
        ----------
        x_grid : array of shape (n_points, dim)
            Grid points
        dt : float
            Time step
            
        Returns
        -------
        K : array of shape (n_points, n_points)
            Transition kernel matrix
        """
        n = len(x_grid)
        var = self.sigma**2 * dt
        
        # Pairwise squared distances
        diff = x_grid[:, np.newaxis, :] - x_grid[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        
        # Gaussian kernel
        K = np.exp(-sq_dist / (2 * var)) / np.sqrt(2 * np.pi * var) ** self.dim
        
        return K
    
    def solve(
        self,
        mu_0: np.ndarray,
        mu_T: np.ndarray,
        x_grid: np.ndarray,
        T: float = 1.0,
        return_potentials: bool = False
    ) -> Dict[str, Any]:
        """
        Solve Schrödinger bridge problem.
        
        Parameters
        ----------
        mu_0 : array of shape (n_0,)
            Initial distribution (weights)
        mu_T : array of shape (n_T,)
            Terminal distribution (weights)
        x_grid : array of shape (n_points, dim)
            State space grid
        T : float
            Total time
        return_potentials : bool
            Return Schrödinger potentials
            
        Returns
        -------
        result : dict
            Transport plan, cost, and optionally potentials
        """
        dt = T / self.n_steps
        
        # Normalize distributions
        mu_0 = mu_0 / (mu_0.sum() + 1e-16)
        mu_T = mu_T / (mu_T.sum() + 1e-16)
        
        # Cost matrix (squared Euclidean distance)
        diff = x_grid[:, np.newaxis, :] - x_grid[np.newaxis, :, :]
        C = np.sum(diff**2, axis=2)
        
        # Solve with Sinkhorn
        cost, P = self.sinkhorn.solve(mu_0, mu_T, C)
        
        result = {
            'transport_plan': P,
            'transport_cost': cost,
            'entropy_regularized_cost': cost
        }
        
        if return_potentials:
            # Extract dual potentials
            # From Sinkhorn iterations
            K = np.exp(-C / self.eps)
            log_K = -C / self.eps
            
            log_u = np.zeros(len(mu_0))
            log_v = np.zeros(len(mu_T))
            
            for _ in range(self.max_iter):
                log_u = np.log(mu_0 + 1e-16) - logsumexp(
                    log_K + log_v[np.newaxis, :], axis=1
                )
                log_v = np.log(mu_T + 1e-16) - logsumexp(
                    log_K.T + log_u[np.newaxis, :], axis=1
                )
            
            result['phi'] = self.eps * log_u  # Initial potential
            result['psi'] = self.eps * log_v  # Terminal potential
        
        return result
    
    def compute_drift(
        self,
        phi: np.ndarray,
        x_grid: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute optimal drift field from Schrödinger potential.
        
        b*(t, x) = σ² ∇ log ψ(t, x)
        
        Parameters
        ----------
        phi : array
            Schrödinger potential
        x_grid : array
            State grid
        t : float
            Current time
            
        Returns
        -------
        b : array
            Optimal drift field
        """
        # Numerical gradient of potential
        if self.dim == 1:
            # 1D case: finite differences
            dx = np.diff(x_grid.flatten())
            dphi = np.diff(phi)
            grad = np.zeros_like(phi)
            grad[1:-1] = (phi[2:] - phi[:-2]) / (x_grid[2:] - x_grid[:-2]).flatten()
            grad[0] = dphi[0] / dx[0]
            grad[-1] = dphi[-1] / dx[-1]
            drift = self.sigma**2 * grad
        else:
            # Multi-dimensional: approximate gradient
            drift = np.zeros_like(x_grid)
            for d in range(self.dim):
                # Sort by coordinate d and compute gradient
                idx = np.argsort(x_grid[:, d])
                sorted_phi = phi[idx]
                sorted_x = x_grid[idx, d]
                grad = np.gradient(sorted_phi, sorted_x)
                drift[idx, d] = self.sigma**2 * grad
        
        return drift
    
    def simulate_bridge(
        self,
        x_0: np.ndarray,
        x_T: np.ndarray,
        n_paths: int = 100,
        T: float = 1.0,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate Schrödinger bridge paths.
        
        Parameters
        ----------
        x_0 : array of shape (dim,)
            Initial point
        x_T : array of shape (dim,)
            Terminal point
        n_paths : int
            Number of paths
        T : float
            Total time
        seed : int, optional
            Random seed
            
        Returns
        -------
        paths : array of shape (n_paths, n_steps+1, dim)
            Simulated bridge paths
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        
        paths = np.zeros((n_paths, self.n_steps + 1, self.dim))
        paths[:, 0, :] = x_0
        
        # Brownian bridge construction
        for path_idx in range(n_paths):
            x = x_0.copy()
            for t_idx in range(self.n_steps):
                t = t_idx * dt
                tau = T - t
                
                if tau > 0:
                    # Brownian bridge drift: (x_T - x) / tau
                    drift = (x_T - x) / tau
                    
                    # Brownian increment
                    dW = np.random.randn(self.dim) * sqrt_dt
                    
                    # Update
                    x = x + drift * dt + self.sigma * dW
                
                paths[path_idx, t_idx + 1, :] = x
        
        return paths


class DiffusionSchrodingerBridge:
    """
    Diffusion Schrödinger Bridge for generative modeling.
    
    Iteratively refines a bridge between noise and data distributions
    using the DSB algorithm (De Bortoli et al., 2021).
    """
    
    def __init__(
        self,
        dim: int,
        sigma: float = 1.0,
        n_steps: int = 100,
        n_iterations: int = 10,
        batch_size: int = 256,
        lr: float = 0.001
    ):
        """
        Initialize DSB.
        
        Parameters
        ----------
        dim : int
            Data dimension
        sigma : float
            Diffusion coefficient
        n_steps : int
            Time discretization steps
        n_iterations : int
            Number of DSB iterations
        batch_size : int
            Training batch size
        lr : float
            Learning rate for drift network
        """
        self.dim = dim
        self.sigma = sigma
        self.n_steps = n_steps
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lr = lr
        
    def fit(
        self,
        data: np.ndarray,
        n_epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Fit DSB to data distribution.
        
        Parameters
        ----------
        data : array of shape (n_samples, dim)
            Training data
        n_epochs : int
            Training epochs per DSB iteration
        verbose : bool
            Print progress
            
        Returns
        -------
        history : dict
            Training history
        """
        n_samples = len(data)
        dt = 1.0 / self.n_steps
        
        history = {'losses': [], 'iterations': []}
        
        # Initialize with standard Brownian bridge
        for iteration in range(self.n_iterations):
            if verbose:
                print(f"DSB Iteration {iteration + 1}/{self.n_iterations}")
            
            # Forward pass: sample from current forward process
            # Backward pass: estimate score/drift
            
            epoch_losses = []
            for epoch in range(n_epochs):
                # Sample batch
                idx = np.random.choice(n_samples, self.batch_size)
                x_0 = data[idx]
                
                # Sample noise
                x_T = np.random.randn(self.batch_size, self.dim) * self.sigma
                
                # Compute loss (simplified: just Brownian bridge)
                loss = self._compute_loss(x_0, x_T, dt)
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            history['losses'].append(avg_loss)
            history['iterations'].append(iteration)
            
            if verbose and iteration % 5 == 0:
                print(f"  Loss: {avg_loss:.6f}")
        
        return history
    
    def _compute_loss(
        self,
        x_0: np.ndarray,
        x_T: np.ndarray,
        dt: float
    ) -> float:
        """Compute bridge matching loss."""
        # Simplified: MSE between endpoints
        # In full implementation, this would be a drift/score matching loss
        return np.mean((x_0 - x_T)**2)
    
    def generate(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate samples from learned bridge.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        seed : int, optional
            Random seed
            
        Returns
        -------
        samples : array of shape (n_samples, dim)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Start from noise
        x = np.random.randn(n_samples, self.dim) * self.sigma
        
        # Simulate backward SDE (simplified: just use mean)
        dt = 1.0 / self.n_steps
        for t_idx in range(self.n_steps):
            # In full implementation, use learned drift
            # Here we just reverse the Brownian motion
            dW = np.random.randn(n_samples, self.dim) * np.sqrt(dt)
            x = x + self.sigma * dW
        
        return x


class MartingaleSchrodingerBridge:
    """
    Martingale Schrödinger Bridge for financial applications.
    
    Extends the standard SB to enforce martingale constraint,
    essential for arbitrage-free pricing models.
    
    Reference:
    - Henry-Labordère (2019): "From (Martingale) Schrödinger bridges
      to a new class of Stochastic Volatility Models"
    """
    
    def __init__(
        self,
        n_assets: int = 1,
        sigma: float = 0.2,
        n_steps: int = 50,
        eps: float = 0.1
    ):
        """
        Initialize Martingale SB.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        sigma : float
            Base volatility
        n_steps : int
            Time steps
        eps : float
            Regularization
        """
        self.n_assets = n_assets
        self.sigma = sigma
        self.n_steps = n_steps
        self.eps = eps
        
    def calibrate_to_vanillas(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        call_prices: np.ndarray,
        n_iter: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Calibrate martingale SB to European option prices.
        
        Parameters
        ----------
        S0 : float
            Spot price
        strikes : array
            Strike prices
        maturities : array
            Option maturities
        call_prices : array
            Market call prices
        n_iter : int
            Calibration iterations
        verbose : bool
            Print progress
            
        Returns
        -------
        result : dict
            Calibrated parameters and fit quality
        """
        n_strikes = len(strikes)
        n_mats = len(maturities)
        
        # Initialize local volatility surface
        vol_surface = np.ones((n_mats, n_strikes)) * self.sigma
        
        history = {'errors': [], 'vols': []}
        
        for iteration in range(n_iter):
            # Price options with current vol surface
            model_prices = self._price_options(S0, strikes, maturities, vol_surface)
            
            # Compute error
            error = np.sqrt(np.mean((model_prices - call_prices)**2))
            history['errors'].append(error)
            history['vols'].append(vol_surface.copy())
            
            # Update vol surface (simplified gradient descent)
            # In full implementation, use adjoint method
            grad = (model_prices - call_prices) * 0.1
            for m, T in enumerate(maturities):
                for k, K in enumerate(strikes):
                    vol_surface[m, k] -= 0.01 * grad[m, k] / (S0 * np.sqrt(T))
                    vol_surface[m, k] = np.clip(vol_surface[m, k], 0.05, 1.0)
            
            if verbose and iteration % 20 == 0:
                print(f"Iteration {iteration}: RMSE = {error:.6f}")
        
        return {
            'vol_surface': vol_surface,
            'final_error': error,
            'history': history
        }
    
    def _price_options(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        vol_surface: np.ndarray
    ) -> np.ndarray:
        """Price options using BS formula with local vol."""
        prices = np.zeros((len(maturities), len(strikes)))
        
        for m, T in enumerate(maturities):
            for k, K in enumerate(strikes):
                sigma = vol_surface[m, k]
                # Black-Scholes formula
                d1 = (np.log(S0 / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                prices[m, k] = S0 * norm.cdf(d1) - K * norm.cdf(d2)
        
        return prices
    
    def simulate_prices(
        self,
        S0: float,
        T: float = 1.0,
        n_paths: int = 10000,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate price paths under calibrated martingale SB.
        
        Parameters
        ----------
        S0 : float
            Initial price
        T : float
            Time horizon
        n_paths : int
            Number of paths
        seed : int, optional
            Random seed
            
        Returns
        -------
        prices : array of shape (n_paths, n_steps+1)
        """
        if seed is not None:
            np.random.seed(seed)
            
        dt = T / self.n_steps
        sqrt_dt = np.sqrt(dt)
        
        prices = np.zeros((n_paths, self.n_steps + 1))
        prices[:, 0] = S0
        
        for t in range(self.n_steps):
            dW = np.random.randn(n_paths) * sqrt_dt
            # Simple GBM (in full implementation, use calibrated drift)
            prices[:, t + 1] = prices[:, t] * np.exp(
                -0.5 * self.sigma**2 * dt + self.sigma * dW
            )
        
        return prices
