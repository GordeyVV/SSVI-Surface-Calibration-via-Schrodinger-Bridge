"""
Nonlinear Schrödinger Equation (NLSE) on Graphs - EIGENSTATE APPROACH
======================================================================

Following core.pdf, the stationary solution ψ(+∞) is an eigenstate of the
effective Hamiltonian H(ψ). We find it by iterative refinement.

Key insight: The stationary state satisfies the nonlinear eigenvalue problem:
    H(ψ)ψ = Eψ
where E is the energy eigenvalue.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from typing import Optional, Tuple, Dict, Any
import warnings


class NLSGroundState:
    """
    Find the ground state (stationary solution) of the NLSE on a graph.
    
    The ground state minimizes the energy functional:
        E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    
    For H = L + diag(|ψ₀|²), this is equivalent to finding the lowest
    eigenstate of a modified Laplacian.
    """
    
    def __init__(
        self,
        n_vertices: int,
        edges: np.ndarray,
        weights: np.ndarray,
        psi0_sq: np.ndarray = None
    ):
        self.n = n_vertices
        self.edges = np.atleast_2d(edges)
        self.weights = np.atleast_1d(weights)
        
        # Initial |ψ₀|² - the "target" distribution
        if psi0_sq is None:
            self.psi0_sq = np.ones(n_vertices) / n_vertices
        else:
            self.psi0_sq = np.asarray(psi0_sq)
        
        self._build_laplacian()
        
    def _build_laplacian(self):
        """Build graph Laplacian matrix."""
        n = self.n
        row, col, data = [], [], []
        
        for idx, (u, v) in enumerate(self.edges):
            w = max(self.weights[idx], 0)
            row.extend([u, v, u, v])
            col.extend([v, u, u, v])
            data.extend([-w, -w, w, w])
        
        self.L = csr_matrix((data, (row, col)), shape=(n, n))
        
        # Full Hamiltonian (dense for eigenvalue computation)
        self.H = self.L.toarray() + np.diag(self.psi0_sq)
        
    def compute_ground_state(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Compute ground state of H.
        
        The ground state is the eigenvector corresponding to the
        smallest eigenvalue of H.
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(self.H)
        
        # Ground state is first eigenvector
        E0 = eigenvalues[0]
        psi0 = eigenvectors[:, 0]
        
        # Ensure normalized
        psi0 = psi0 / np.linalg.norm(psi0)
        
        # Make sure phase is consistent (largest component positive)
        idx_max = np.argmax(np.abs(psi0))
        if psi0[idx_max] < 0:
            psi0 = -psi0
        
        if verbose:
            print(f"Ground state energy: {E0:.4f}")
            print(f"First 5 eigenvalues: {eigenvalues[:5]}")
            print(f"Ground state norm: {np.linalg.norm(psi0):.6f}")
        
        return {
            'psi_ground': psi0,
            'energy': E0,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }
    
    def compute_excited_states(self, n_states: int = 5, verbose: bool = False) -> Dict[str, Any]:
        """
        Compute first n_states eigenstates of H.
        """
        eigenvalues, eigenvectors = eigh(self.H)
        
        states = []
        for i in range(min(n_states, self.n)):
            psi = eigenvectors[:, i]
            psi = psi / np.linalg.norm(psi)
            states.append({
                'psi': psi,
                'energy': eigenvalues[i],
                'index': i
            })
        
        if verbose:
            print(f"Computed {len(states)} eigenstates")
            for s in states:
                print(f"  State {s['index']}: E = {s['energy']:.4f}")
        
        return {
            'states': states,
            'eigenvalues': eigenvalues[:len(states)],
            'eigenvectors': eigenvectors[:, :len(states)]
        }
    
    def update_weights(self, new_weights: np.ndarray):
        """Update edge weights and rebuild Laplacian."""
        self.weights = np.maximum(new_weights, 0)
        self._build_laplacian()
    
    def update_psi0_sq(self, new_psi0_sq: np.ndarray):
        """Update the |ψ₀|² distribution."""
        self.psi0_sq = np.asarray(new_psi0_sq)
        self.psi0_sq = np.maximum(self.psi0_sq, 0)
        self._build_laplacian()
    
    def compute_sobolev_norm(self, psi: np.ndarray) -> float:
        """
        Compute H¹ Sobolev norm:
        ||ψ||²_{H¹} = ||ψ||² + ||∇ψ||²
        
        For graphs: ||ψ||²_{H¹} = Σ|ψ|² + Σ_{edges} w|ψ_u - ψ_v|²
        """
        l2_sq = np.sum(np.abs(psi)**2)
        
        h1_sq = l2_sq
        for idx, (u, v) in enumerate(self.edges):
            w = self.weights[idx]
            h1_sq += w * np.abs(psi[u] - psi[v])**2
        
        return np.sqrt(h1_sq)
    
    def compute_density(self, psi: np.ndarray) -> np.ndarray:
        """Compute probability density ρ = |ψ|²."""
        return np.abs(psi)**2


class FANNLayerCorrected:
    """
    FANN Layer with proper ground state computation.
    
    Following core.pdf, the layer:
    1. Takes input x and converts to initial distribution ρ₀ = |ψ₀|²
    2. Computes ground state of H = L + diag(ρ₀)
    3. Returns ground state ψ_ground
    
    The graph weights are learned to minimize a loss function.
    """
    
    def __init__(
        self,
        n_vertices: int,
        edges: np.ndarray,
        weights: np.ndarray = None
    ):
        self.n = n_vertices
        self.edges = np.atleast_2d(edges)
        
        if weights is None:
            self.weights = np.ones(len(self.edges))
        else:
            self.weights = np.atleast_1d(weights)
        
        self.solver = NLSGroundState(self.n, self.edges, self.weights)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: x → ground state.
        
        x encodes the initial distribution ρ₀.
        """
        x = np.asarray(x).flatten()[:self.n]
        if len(x) < self.n:
            x = np.concatenate([x, np.zeros(self.n - len(x))])
        
        # Convert to distribution (ensure positive and normalized)
        psi0_sq = np.abs(x)**2
        if np.sum(psi0_sq) > 0:
            psi0_sq = psi0_sq / np.sum(psi0_sq)
        else:
            psi0_sq = np.ones(self.n) / self.n
        
        # Update Hamiltonian with new initial distribution
        self.solver.update_psi0_sq(psi0_sq)
        
        # Compute ground state
        result = self.solver.compute_ground_state(verbose=False)
        
        return result['psi_ground']
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_epochs: int = 50,
        lr: float = 0.01,
        l1_reg: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Fit graph weights to match inputs to outputs.
        
        Parameters:
            X: Input data (n_samples x n_features)
            Y: Target outputs (n_samples x n_features)
            n_epochs: Training epochs
            lr: Learning rate
            l1_reg: L1 regularization for sparsity
            verbose: Print progress
        """
        history = {'losses': [], 'weights': []}
        
        def loss_fn(psi, target):
            # Correlation loss (allowing for phase)
            correlation = np.abs(np.vdot(psi, target))
            return 1 - correlation  # Minimize 1 - |correlation|
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            grads = np.zeros(len(self.weights))
            eps = 1e-4
            
            for i in range(len(X)):
                # Forward pass
                psi = self.forward(X[i])
                loss = loss_fn(psi, Y[i])
                total_loss += loss
                
                # Numerical gradient
                for j in range(len(self.weights)):
                    w_orig = self.weights[j]
                    
                    # Plus
                    self.weights[j] = w_orig + eps
                    self.solver.update_weights(self.weights)
                    psi_plus = self.forward(X[i])
                    loss_plus = loss_fn(psi_plus, Y[i])
                    
                    # Minus
                    self.weights[j] = w_orig - eps
                    self.solver.update_weights(self.weights)
                    psi_minus = self.forward(X[i])
                    loss_minus = loss_fn(psi_minus, Y[i])
                    
                    grads[j] += (loss_plus - loss_minus) / (2 * eps)
                    self.weights[j] = w_orig
            
            # L1 regularization
            grads += l1_reg * np.sign(self.weights)
            
            # Update weights
            self.weights -= lr * grads
            self.weights = np.maximum(self.weights, 0)
            self.solver.update_weights(self.weights)
            
            avg_loss = total_loss / len(X)
            history['losses'].append(avg_loss)
            history['weights'].append(self.weights.copy())
            
            if verbose and epoch % 10 == 0:
                n_active = np.sum(self.weights > 0.01)
                print(f"Epoch {epoch}: Loss={avg_loss:.6f}, Active edges={n_active}")
        
        return history


def create_random_graph(n_vertices: int, edge_prob: float = 0.3, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create random Erdős-Rényi graph."""
    if seed is not None:
        np.random.seed(seed)
    
    edges = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if np.random.rand() < edge_prob:
                edges.append([i, j])
    
    if len(edges) == 0:
        # Ensure at least one edge
        edges.append([0, min(1, n_vertices - 1)])
    
    return np.array(edges), np.ones(len(edges))


def create_knn_graph(points: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Create k-NN graph from point cloud."""
    from scipy.spatial.distance import cdist
    
    n = len(points)
    D = cdist(points, points)
    
    edges = []
    weights = []
    
    for i in range(n):
        neighbors = np.argsort(D[i])[1:k+1]
        sigma = np.mean(D[i, neighbors])
        
        for j in neighbors:
            if i < j:
                edges.append([i, j])
                weights.append(np.exp(-D[i, j]**2 / (2 * sigma**2)))
    
    return np.array(edges), np.array(weights)


if __name__ == "__main__":
    print("Testing NLSE Ground State Solver")
    print("=" * 50)
    
    # Create graph
    n = 20
    edges, weights = create_random_graph(n, edge_prob=0.3, seed=42)
    
    print(f"Graph: {n} vertices, {len(edges)} edges")
    
    # Initialize solver
    solver = NLSGroundState(n, edges, weights)
    
    # Random initial distribution
    np.random.seed(42)
    psi0_sq = np.random.rand(n)
    psi0_sq = psi0_sq / psi0_sq.sum()
    
    solver.update_psi0_sq(psi0_sq)
    
    # Compute ground state
    result = solver.compute_ground_state(verbose=True)
    
    # Verify properties
    psi = result['psi_ground']
    print(f"\nGround state norm: {np.linalg.norm(psi):.6f}")
    print(f"Sobolev H¹ norm: {solver.compute_sobolev_norm(psi):.6f}")
    
    # Verify eigenvalue equation
    H_psi = solver.H @ psi
    E_psi = result['energy'] * psi
    print(f"\nEigenvalue equation residual: {np.linalg.norm(H_psi - E_psi):.2e}")
    
    # Test FANN layer
    print("\n" + "=" * 50)
    print("Testing FANN Layer")
    print("=" * 50)
    
    layer = FANNLayerCorrected(n, edges, weights)
    
    # Create some data
    X = np.random.randn(10, n)
    Y = np.random.randn(10, n)
    
    # Normalize
    for i in range(len(X)):
        X[i] = X[i] / np.linalg.norm(X[i])
        Y[i] = Y[i] / np.linalg.norm(Y[i])
    
    history = layer.fit(X, Y, n_epochs=30, verbose=True)
