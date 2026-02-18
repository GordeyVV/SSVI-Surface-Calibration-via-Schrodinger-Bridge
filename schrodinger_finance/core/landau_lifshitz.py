"""
Landau-Lifshitz Equation Solver
================================

Gauge-equivalent formulation of NLSE using spin dynamics.

The Landau-Lifshitz equation:
    dS⃗/dt = S⃗ × (-2 Σ_k w_{jk} S⃗_k + 2|ψ_j(0)|² e₃) - γS⃗ × (S⃗ × D⃗_j)

This provides a different perspective on the same physics,
with S⃗ representing spin alignment on the Bloch sphere.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from scipy.integrate import solve_ivp

from .graph_sobolev import GraphSobolevSpace, SpinFieldSpace


class LandauLifshitzSolver:
    """
    Solver for the Landau-Lifshitz equation on graphs.
    
    Gauge-equivalent to NLSE via stereographic projection.
    """
    
    def __init__(
        self,
        graph: GraphSobolevSpace,
        gamma: float = 0.1,
        dt: float = 0.01
    ):
        """
        Initialize LL solver.
        
        Parameters
        ----------
        graph : GraphSobolevSpace
            Base graph
        gamma : float
            Damping parameter
        dt : float
            Time step
        """
        self.graph = graph
        self.spin_space = SpinFieldSpace(graph)
        self.n_vertices = graph.n_vertices
        self.gamma = gamma
        self.dt = dt
        
    def effective_field(
        self,
        S: np.ndarray,
        S0: np.ndarray
    ) -> np.ndarray:
        """
        Compute effective field for each spin.
        
        H_eff = -2 Σ_k w_{jk} S⃗_k + 2|ψ_j(0)|² e₃
        
        Parameters
        ----------
        S : array of shape (n_vertices, 3)
            Current spin field
        S0 : array of shape (n_vertices, 3)
            Initial spin field
            
        Returns
        -------
        H_eff : array of shape (n_vertices, 3)
        """
        H_eff = np.zeros_like(S)
        
        e3 = np.array([0, 0, 1])
        
        for idx, (u, v) in enumerate(self.graph.edges):
            w = self.graph.edge_weights[idx]
            # Contribution from neighbor
            H_eff[u] -= 2 * w * S[v]
            H_eff[v] -= 2 * w * S[u]
        
        # Add initial condition term
        psi0_sq = (1 - S0[:, 2]) / (1 + S0[:, 2] + 1e-10)  # |ψ_0|² from S0
        H_eff += 2 * psi0_sq[:, np.newaxis] * e3
        
        return H_eff
    
    def dissipation_field(
        self,
        S: np.ndarray,
        S0: np.ndarray
    ) -> np.ndarray:
        """
        Compute dissipation field D⃗ for LL equation.
        """
        D = np.zeros_like(S)
        
        for idx, (u, v) in enumerate(self.graph.edges):
            w = self.graph.edge_weights[idx]
            D[u] -= 2 * w * (S[v] - S[u])
            D[v] -= 2 * w * (S[u] - S[v])
        
        return D
    
    def rhs(self, t: float, S_flat: np.ndarray, S0: np.ndarray) -> np.ndarray:
        """
        Right-hand side of LL equation (flattened for ODE solver).
        
        dS⃗/dt = S⃗ × H_eff - γS⃗ × (S⃗ × D⃗)
        """
        S = S_flat.reshape(self.n_vertices, 3)
        
        H_eff = self.effective_field(S, S0)
        D = self.dissipation_field(S, S0)
        
        # Precession term
        precession = np.cross(S, H_eff)
        
        # Damping term (Gilbert damping)
        damping = -self.gamma * np.cross(S, np.cross(S, D))
        
        dS_dt = precession + damping
        
        return dS_dt.flatten()
    
    def solve(
        self,
        S0: np.ndarray,
        t_span: Tuple[float, float] = (0, 50),
        n_points: int = 500
    ) -> Dict[str, Any]:
        """
        Solve LL equation.
        
        Parameters
        ----------
        S0 : array of shape (n_vertices, 3)
            Initial spin configuration
        t_span : tuple
            Time interval
        n_points : int
            Number of output points
            
        Returns
        -------
        Solution dictionary
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        
        result = solve_ivp(
            lambda t, y: self.rhs(t, y, S0),
            t_span,
            S0.flatten(),
            t_eval=t_eval,
            method='RK45'
        )
        
        S_trajectory = result.y.T.reshape(-1, self.n_vertices, 3)
        
        return {
            'S_final': S_trajectory[-1],
            'S_trajectory': S_trajectory,
            'times': result.t,
            'converged': result.success
        }
    
    def from_wavefunction(self, psi: np.ndarray) -> np.ndarray:
        """Convert wavefunction to spin field."""
        return self.spin_space.stereographic_projection(psi)
    
    def to_wavefunction(self, S: np.ndarray) -> np.ndarray:
        """Convert spin field to wavefunction."""
        return self.spin_space.inverse_stereographic(S)
