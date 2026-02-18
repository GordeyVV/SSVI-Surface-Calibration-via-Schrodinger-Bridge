"""
Graph Moduli Space Optimization
===============================

Optimization on the space of weighted graphs with:
- Topological correctness (homology recovery)
- Metric convergence (Gromov-Hausdorff)
- Efficient stratified search
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, List, Tuple
from scipy.sparse import csr_matrix
import warnings

from .graph_sobolev import GraphSobolevSpace


class ModuliSpaceOptimizer:
    """
    Optimization on the moduli space of weighted graphs.
    
    The moduli space M consists of all configurations (E, w) where:
    - E ⊂ V × V is the edge set
    - w : E → R⁺ are edge weights
    
    M is stratified: each edge set E defines a stratum M(E) ≃ R⁺^{|E|}
    """
    
    def __init__(
        self,
        n_vertices: int,
        loss_fn: Callable,
        theta: float = 0.01,
        Theta: float = 0.1,
        lr: float = 0.01,
        l1_reg: float = 0.001,
        l2_reg: float = 0.0001
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        n_vertices : int
            Number of vertices
        loss_fn : callable
            Loss function L(psi, w) → scalar
        theta : float
            Threshold for edge deletion
        Theta : float
            Threshold for edge addition
        lr : float
            Learning rate
        l1_reg : float
            L1 regularization (sparsity)
        l2_reg : float
            L2 regularization (smoothness)
        """
        self.n_vertices = n_vertices
        self.loss_fn = loss_fn
        self.theta = theta
        self.Theta = Theta
        self.lr = lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        # Initialize empty graph
        self.edges = []
        self.weights = np.array([])
        
    def initialize_random(
        self,
        edge_prob: float = 0.3,
        seed: Optional[int] = None
    ):
        """Initialize with random graph."""
        if seed is not None:
            np.random.seed(seed)
            
        self.edges = []
        self.weights = []
        
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if np.random.random() < edge_prob:
                    self.edges.append([i, j])
                    self.weights.append(1.0)
        
        self.edges = np.array(self.edges) if self.edges else np.array([]).reshape(0, 2)
        self.weights = np.array(self.weights)
        
    def initialize_knn(
        self,
        points: np.ndarray,
        k: int = 5
    ):
        """Initialize with k-NN graph from point cloud."""
        from scipy.spatial.distance import cdist
        
        distances = cdist(points, points)
        self.edges = []
        self.weights = []
        
        for i in range(self.n_vertices):
            neighbors = np.argsort(distances[i])[1:k+1]
            for j in neighbors:
                if i < j:
                    self.edges.append([i, j])
                    sigma = np.median(distances[i][neighbors])
                    w = np.exp(-distances[i, j]**2 / (2 * sigma**2))
                    self.weights.append(w)
        
        self.edges = np.array(self.edges) if self.edges else np.array([]).reshape(0, 2)
        self.weights = np.array(self.weights)
        
    def get_graph(self) -> GraphSobolevSpace:
        """Get current graph as GraphSobolevSpace."""
        if len(self.edges) == 0:
            # Return trivial graph with single vertex
            return GraphSobolevSpace(1, np.array([[0, 0]]))
        return GraphSobolevSpace(
            self.n_vertices, 
            self.edges, 
            edge_weights=self.weights
        )
    
    def compute_gradient(
        self,
        psi: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient w.r.t. edge weights."""
        if len(self.weights) == 0:
            return np.array([])
            
        grads = np.zeros(len(self.weights))
        
        for e in range(len(self.weights)):
            # Numerical gradient
            w_plus = self.weights.copy()
            w_plus[e] += eps
            loss_plus = self.loss_fn(psi, w_plus)
            
            w_minus = self.weights.copy()
            w_minus[e] -= eps
            loss_minus = self.loss_fn(psi, w_minus)
            
            grads[e] = (loss_plus - loss_minus) / (2 * eps)
        
        return grads
    
    def step(
        self,
        psi: np.ndarray,
        verbose: bool = False
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Perform one optimization step.
        
        Returns
        -------
        loss : float
            Current loss value
        info : dict
            Step information
        """
        if len(self.edges) == 0:
            # Add edges with probability
            self._add_initial_edges()
            
        # Compute current loss
        loss = self.loss_fn(psi, self.weights)
        
        # Compute gradients
        grads = self.compute_gradient(psi)
        
        # Add regularization
        grads += self.l1_reg * np.sign(self.weights)
        grads += self.l2_reg * self.weights
        
        # Update weights
        self.weights -= self.lr * grads
        
        # Project to non-negative
        self.weights = np.maximum(self.weights, 0)
        
        # Edge deletion
        keep_mask = self.weights >= self.theta
        n_deleted = np.sum(~keep_mask)
        
        if np.any(~keep_mask):
            self.edges = self.edges[keep_mask]
            self.weights = self.weights[keep_mask]
        
        # Edge addition (check candidate edges)
        n_added = self._check_edge_addition(psi, grads)
        
        info = {
            'loss': loss,
            'n_edges': len(self.edges),
            'n_deleted': n_deleted,
            'n_added': n_added,
            'max_weight': np.max(self.weights) if len(self.weights) > 0 else 0,
            'min_weight': np.min(self.weights) if len(self.weights) > 0 else 0
        }
        
        if verbose:
            print(f"Loss: {loss:.6f}, Edges: {len(self.edges)}, "
                  f"Added: {n_added}, Deleted: {n_deleted}")
        
        return loss, info
    
    def _add_initial_edges(self):
        """Add initial edges when graph is empty."""
        # Add a minimal spanning set of edges
        for i in range(min(3, self.n_vertices - 1)):
            self.edges.append([i, i + 1])
            self.weights.append(1.0)
        self.edges = np.array(self.edges)
        self.weights = np.array(self.weights)
        
    def _check_edge_addition(
        self,
        psi: np.ndarray,
        current_grads: np.ndarray
    ) -> int:
        """Check if any candidate edges should be added."""
        if len(self.edges) == 0:
            return 0
            
        # Current edge set
        edge_set = set(map(tuple, self.edges))
        
        n_added = 0
        new_edges = list(self.edges)
        new_weights = list(self.weights)
        
        # Sample candidate edges
        for i in range(self.n_vertices):
            for j in range(i + 1, self.n_vertices):
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    # Test gradient for this edge
                    test_weights = np.append(self.weights, self.theta)
                    test_loss = self.loss_fn(psi, test_weights)
                    
                    if test_loss < -self.Theta:
                        new_edges.append([i, j])
                        new_weights.append(2 * self.theta)
                        n_added += 1
        
        if n_added > 0:
            self.edges = np.array(new_edges)
            self.weights = np.array(new_weights)
        
        return n_added
    
    def optimize(
        self,
        psi_fn: Callable,
        n_epochs: int = 100,
        verbose: bool = True,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Run full optimization.
        
        Parameters
        ----------
        psi_fn : callable
            Function to compute psi given graph (returns stationary state)
        n_epochs : int
            Maximum epochs
        verbose : bool
            Print progress
        patience : int
            Early stopping patience
            
        Returns
        -------
        Training history
        """
        history = {
            'losses': [],
            'n_edges': [],
            'weights': []
        }
        
        best_loss = np.inf
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Get current graph
            graph = self.get_graph()
            
            # Compute stationary state
            psi = psi_fn(graph)
            
            # Optimization step
            loss, info = self.step(psi, verbose=False)
            
            history['losses'].append(loss)
            history['n_edges'].append(info['n_edges'])
            history['weights'].append(self.weights.copy())
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
                
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.6f}, Edges={info['n_edges']}")
        
        return history
    
    def compute_homology(self) -> Dict[str, int]:
        """Compute Betti numbers (β₀, β₁) of current graph."""
        if len(self.edges) == 0:
            return {'beta_0': self.n_vertices, 'beta_1': 0}
        
        # Connected components (β₀)
        visited = np.zeros(self.n_vertices, dtype=bool)
        components = 0
        
        adjacency = {}
        for idx, (u, v) in enumerate(self.edges):
            if self.weights[idx] > self.theta:
                if u not in adjacency:
                    adjacency[u] = []
                if v not in adjacency:
                    adjacency[v] = []
                adjacency[u].append(v)
                adjacency[v].append(u)
        
        def dfs(node):
            stack = [node]
            while stack:
                n = stack.pop()
                if not visited[n]:
                    visited[n] = True
                    if n in adjacency:
                        stack.extend(adjacency[n])
        
        for v in range(self.n_vertices):
            if not visited[v]:
                dfs(v)
                components += 1
        
        beta_0 = components
        
        # Euler characteristic for β₁
        # χ = V - E + F = β₀ - β₁ (for planar graphs)
        # For general graphs: β₁ = E - V + β₀
        n_active_edges = np.sum(self.weights > self.theta)
        beta_1 = max(0, n_active_edges - self.n_vertices + beta_0)
        
        return {'beta_0': beta_0, 'beta_1': beta_1}
