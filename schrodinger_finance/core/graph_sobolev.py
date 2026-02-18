"""
Sobolev Spaces on Graphs
========================

Implementation of discrete Sobolev spaces H¹(V), H²(V) on weighted graphs
following the FANN framework (Pasechnyuk-Vilensky & Takáč, 2025).

Mathematical Foundation:
- L²(V): functions f: V → C with norm ||f||²_L² = Σ_v |f(v)|² μ(v)
- H¹(V): functions with finite H¹ norm including gradient term
- H²(V): functions with Laplacian in L²

Key operators:
- Discrete gradient d: L²(V) → L²(E)
- Laplacian Δ: L²(V) → L²(V)
"""

import numpy as np
from scipy import sparse
from typing import Optional, Tuple, Callable
import warnings


class GraphSobolevSpace:
    """
    Sobolev spaces on weighted graphs with full functional analysis.
    
    Parameters
    ----------
    n_vertices : int
        Number of vertices in the graph
    edges : array-like of shape (n_edges, 2)
        Edge connections as pairs of vertex indices
    vertex_weights : array-like of shape (n_vertices,), optional
        Vertex measures μ(v), default is uniform (μ(v) = 1)
    edge_weights : array-like of shape (n_edges,), optional
        Edge measures ρ(e), default is uniform (ρ(e) = 1)
    """
    
    def __init__(
        self,
        n_vertices: int,
        edges: np.ndarray,
        vertex_weights: Optional[np.ndarray] = None,
        edge_weights: Optional[np.ndarray] = None
    ):
        self.n_vertices = n_vertices
        self.edges = np.asarray(edges)
        self.n_edges = len(edges)
        
        # Set default weights
        self.vertex_weights = (
            np.ones(n_vertices) if vertex_weights is None 
            else np.asarray(vertex_weights)
        )
        self.edge_weights = (
            np.ones(self.n_edges) if edge_weights is None 
            else np.asarray(edge_weights)
        )
        
        # Compute derived quantities
        self._compute_degree()
        self._build_laplacian()
        self._build_gradient()
        
    def _compute_degree(self):
        """Compute vertex degrees."""
        self.degrees = np.zeros(self.n_vertices)
        for idx, (u, v) in enumerate(self.edges):
            self.degrees[u] += self.edge_weights[idx]
            self.degrees[v] += self.edge_weights[idx]
        self.max_degree = np.max(self.degrees)
        
    def _build_laplacian(self):
        """
        Build sparse Laplacian matrix.
        
        The discrete Laplacian is defined as:
        (Δf)(v) = (1/μ(v)) Σ_{e=(v,u)} ρ(e)(f(u) - f(v))
        """
        row = []
        col = []
        data = []
        
        for idx, (u, v) in enumerate(self.edges):
            w = self.edge_weights[idx]
            
            # Diagonal contributions
            row.extend([u, v])
            col.extend([u, v])
            data.extend([-w / self.vertex_weights[u], -w / self.vertex_weights[v]])
            
            # Off-diagonal contributions
            row.extend([u, v])
            col.extend([v, u])
            data.extend([w / self.vertex_weights[u], w / self.vertex_weights[v]])
        
        self.laplacian = sparse.csr_matrix(
            (data, (row, col)), 
            shape=(self.n_vertices, self.n_vertices)
        )
        
    def _build_gradient(self):
        """
        Build gradient operator as sparse matrix.
        
        The gradient maps vertex functions to edge functions:
        (df)(e) = f(u) - f(v) for oriented edge e = (v, u)
        """
        row = []
        col = []
        data = []
        
        for idx, (u, v) in enumerate(self.edges):
            # Gradient: d_ij = δ_{i} - δ_{j}
            row.extend([idx, idx])
            col.extend([u, v])
            data.extend([1, -1])
        
        self.gradient = sparse.csr_matrix(
            (data, (row, col)),
            shape=(self.n_edges, self.n_vertices)
        )
        self.gradient_adjoint = self.gradient.T  # Divergence operator
        
    def l2_norm(self, f: np.ndarray) -> float:
        """
        Compute L² norm of vertex function.
        
        ||f||²_L² = Σ_v |f(v)|² μ(v)
        """
        return np.sqrt(np.sum(np.abs(f)**2 * self.vertex_weights))
    
    def h1_norm(self, f: np.ndarray) -> float:
        """
        Compute H¹ norm of vertex function.
        
        ||f||²_H¹ = ||f||²_L² + ||∇f||²_L²(E)
        """
        l2_sq = np.sum(np.abs(f)**2 * self.vertex_weights)
        
        # Gradient norm
        grad_f = self.gradient @ f
        grad_sq = np.sum(np.abs(grad_f)**2 * self.edge_weights)
        
        return np.sqrt(l2_sq + grad_sq)
    
    def h2_norm(self, f: np.ndarray) -> float:
        """
        Compute H² norm of vertex function.
        
        ||f||²_H² = ||f||²_L² + ||Δf||²_L²
        """
        l2_sq = np.sum(np.abs(f)**2 * self.vertex_weights)
        
        # Laplacian norm
        lap_f = self.laplacian @ f
        lap_sq = np.sum(np.abs(lap_f)**2 * self.vertex_weights)
        
        return np.sqrt(l2_sq + lap_sq)
    
    def l2_inner_product(self, f: np.ndarray, g: np.ndarray) -> complex:
        """
        Compute L² inner product.
        
        ⟨f, g⟩_L² = Σ_v f(v) ḡ(v) μ(v)
        """
        return np.sum(f * np.conj(g) * self.vertex_weights)
    
    def h1_inner_product(self, f: np.ndarray, g: np.ndarray) -> complex:
        """
        Compute H¹ inner product.
        
        ⟨f, g⟩_H¹ = ⟨f, g⟩_L² + ⟨df, dg⟩_L²(E)
        """
        l2_ip = self.l2_inner_product(f, g)
        
        df = self.gradient @ f
        dg = self.gradient @ g
        grad_ip = np.sum(df * np.conj(dg) * self.edge_weights)
        
        return l2_ip + grad_ip
    
    def dirichlet_energy(self, f: np.ndarray) -> float:
        """
        Compute Dirichlet energy.
        
        E(f) = ½ Σ_{e=(u,v)} |f(u) - f(v)|² ρ(e)
        """
        grad_f = self.gradient @ f
        return 0.5 * np.sum(np.abs(grad_f)**2 * self.edge_weights)
    
    def spectral_decomposition(self, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of Laplacian.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute (default: all)
            
        Returns
        -------
        eigenvalues : array
            Laplacian eigenvalues (sorted ascending)
        eigenvectors : array
            Corresponding eigenvectors as columns
        """
        if k is None or k >= self.n_vertices:
            # Full decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian.toarray())
        else:
            # Partial decomposition using sparse solver
            eigenvalues, eigenvectors = sparse.linalg.eigsh(
                self.laplacian, k=k, which='SM'
            )
        
        return eigenvalues, eigenvectors
    
    def heat_kernel(self, t: float, f: np.ndarray) -> np.ndarray:
        """
        Apply heat kernel: exp(tΔ) f
        
        Parameters
        ----------
        t : float
            Time parameter (diffusion time)
        f : array
            Input function
            
        Returns
        -------
        Heat kernel applied to f
        """
        # Use spectral decomposition for heat kernel
        eigenvalues, eigenvectors = self.spectral_decomposition()
        
        # Project onto eigenvectors
        coeffs = eigenvectors.T @ (f * self.vertex_weights)
        
        # Apply exponential decay
        coeffs *= np.exp(-t * eigenvalues)
        
        # Reconstruct
        return eigenvectors @ coeffs
    
    def build_random_graph(
        n_vertices: int,
        edge_probability: float = 0.3,
        seed: Optional[int] = None
    ) -> 'GraphSobolevSpace':
        """
        Create a random Erdős-Rényi graph.
        
        Parameters
        ----------
        n_vertices : int
            Number of vertices
        edge_probability : float
            Probability of edge between any two vertices
        seed : int, optional
            Random seed
            
        Returns
        -------
        GraphSobolevSpace instance
        """
        if seed is not None:
            np.random.seed(seed)
            
        edges = []
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if np.random.random() < edge_probability:
                    edges.append([i, j])
                    
        return GraphSobolevSpace(n_vertices, np.array(edges))
    
    def build_knn_graph(
        points: np.ndarray,
        k: int = 5,
        weighted: bool = True
    ) -> 'GraphSobolevSpace':
        """
        Build k-nearest neighbor graph from point cloud.
        
        Parameters
        ----------
        points : array of shape (n_points, dim)
            Point cloud data
        k : int
            Number of nearest neighbors
        weighted : bool
            If True, use Gaussian kernel weights
            
        Returns
        -------
        GraphSobolevSpace instance
        """
        from scipy.spatial.distance import cdist
        
        n_points = len(points)
        distances = cdist(points, points)
        
        edges = []
        weights = []
        
        for i in range(n_points):
            # Find k nearest neighbors (excluding self)
            neighbors = np.argsort(distances[i])[1:k+1]
            
            for j in neighbors:
                if i < j:  # Avoid duplicate edges
                    edges.append([i, j])
                    if weighted:
                        # Gaussian kernel weight
                        sigma = np.median(distances[i][neighbors])
                        weights.append(np.exp(-distances[i, j]**2 / (2 * sigma**2)))
                    else:
                        weights.append(1.0)
        
        graph = GraphSobolevSpace(n_points, np.array(edges), edge_weights=np.array(weights))
        return graph


class SpinFieldSpace:
    """
    Sobolev space for spin fields S⃗ : V → R³ with |S⃗| = 1.
    
    H¹(V; R³) for vector-valued spin fields.
    """
    
    def __init__(self, graph_space: GraphSobolevSpace):
        """
        Initialize spin field space based on graph.
        
        Parameters
        ----------
        graph_space : GraphSobolevSpace
            Base graph Sobolev space
        """
        self.graph = graph_space
        self.n_vertices = graph_space.n_vertices
        
    def stereographic_projection(self, psi: np.ndarray) -> np.ndarray:
        """
        Map complex wavefunction to spin field via stereographic projection.
        
        S⃗_j = ( (ψ_j + ψ̄_j) / (1 + |ψ_j|²),
                 i(ψ̄_j - ψ_j) / (1 + |ψ_j|²),
                 (1 - |ψ_j|²) / (1 + |ψ_j|²) )
                 
        Parameters
        ----------
        psi : array of shape (n_vertices,)
            Complex wavefunction
            
        Returns
        -------
        S : array of shape (n_vertices, 3)
            Spin field with |S| = 1
        """
        psi_abs_sq = np.abs(psi)**2
        denom = 1 + psi_abs_sq
        
        S = np.zeros((self.n_vertices, 3), dtype=float)
        S[:, 0] = 2 * np.real(psi) / denom
        S[:, 1] = 2 * np.imag(psi) / denom
        S[:, 2] = (1 - psi_abs_sq) / denom
        
        return S
    
    def inverse_stereographic(self, S: np.ndarray) -> np.ndarray:
        """
        Inverse stereographic projection: spin field to wavefunction.
        
        ψ_j = (S_j^x + i S_j^y) / (1 + S_j^z)
        
        Parameters
        ----------
        S : array of shape (n_vertices, 3)
            Spin field
            
        Returns
        -------
        psi : array of shape (n_vertices,)
            Complex wavefunction
        """
        psi = (S[:, 0] + 1j * S[:, 1]) / (1 + S[:, 2] + 1e-10)
        return psi
    
    def spin_field_norm(self, S: np.ndarray) -> float:
        """
        H¹ norm for spin field.
        
        ||S||²_H¹ = Σ_v ||S(v)||² μ(v) + Σ_e ||S(u) - S(v)||² ρ(e)
        """
        # Vertex contribution
        vertex_norm = np.sum(np.linalg.norm(S, axis=1)**2 * self.graph.vertex_weights)
        
        # Edge contribution
        edge_norm = 0.0
        for idx, (u, v) in enumerate(self.graph.edges):
            edge_norm += np.linalg.norm(S[u] - S[v])**2 * self.graph.edge_weights[idx]
        
        return np.sqrt(vertex_norm + edge_norm)
