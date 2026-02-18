# SSVI Surface Calibration via Schrödinger Bridge

---

## Project Structure

```
/SSVI-Surface-Calibration-via-Schrodinger-Bridge/
│
├── 📁 schrodinger_finance/                 # Core Python Package
│   ├── __init__.py
│   ├── demo.py                             # Demonstration script
│   │
│   ├── 📁 core/                            # Mathematical Core
│   │   ├── __init__.py
│   │   ├── graph_sobolev.py               # Graph Sobolev spaces H¹(V), H²(V)
│   │   ├── nlse_solver.py                 # Nonlinear Schrödinger Equation solver
│   │   ├── landau_lifshitz.py             # Landau-Lifshitz dynamics
│   │   └── moduli_space.py                # Moduli space of connections
│   │
│   ├── 📁 optimal_transport/               # OT & Schrödinger Bridge
│   │   ├── __init__.py
│   │   ├── sinkhorn.py                    # Sinkhorn algorithm implementation
│   │   ├── schrodinger_bridge.py          # Schrödinger bridge solver
│   │   └── hjb_solver.py                  # Hamilton-Jacobi-Bellman solver
│   │
│   ├── 📁 finance/                         # Financial Applications
│   │   ├── __init__.py
│   │   ├── heston_model.py                # Heston stochastic volatility model
│   │   ├── option_pricing.py              # Black-Scholes pricing functions
│   │   ├── calibration.py                 # Model calibration framework
│   │   └── volatility_surface.py          # Volatility surface construction
│   │
│   └── 📁 data/
│       └── __init__.py                    # Data handling utilities
│
├── 📄 ssvi_calibration.py                  # SSVI Calibration Script
├── 📄 ot_calibration.py                    # Enhanced OT Calibration Script
├── 📄 run_demo.py                          # Package demonstration runner
├── 📄 generate_report.py                   # Report generation script
│
├── 📊 Generated Plots
│   │
│   ├── 🖼️ SSVI Surface & Calibration
│   │   ├── ssvi_surface.png               # 3D SSVI implied volatility surface
│   │   ├── volatility_smiles.png          # Volatility smiles at different maturities
│   │   ├── calibration_results.png        # Market vs model IV comparison
│   │   └── term_structure.png             # IV term structure
│   │
│   ├── 🖼️ OT Calibration Results
│   │   ├── ot_ssvi_surface.png            # Schrödinger Bridge calibrated surface
│   │   ├── ot_calibration_fit.png         # Detailed fit analysis (4 panels)
│   │   └── ot_option_prices.png           # Option prices and Greeks
│   │
│   └── 🖼️ Core Module Demos
│       ├── demo_graph_sobolev.png         # Graph Sobolev space visualization
│       ├── demo_ground_state.png          # NLSE ground state
│       ├── demo_optimal_transport.png     # Optimal transport visualization
│       ├── demo_heston.png                # Heston model simulation
│       ├── demo_heston_model.png          # Heston calibration
│       ├── demo_calibration.png           # Calibration demo
│       └── demo_summary.png               # Summary dashboard
│
└── README.md                          # Project README
```

---

## 1. Theoretical Framework

### 1.1 SSVI Parametrization

The Surface Stochastic Volatility Implied (SSVI) formula provides an **arbitrage-free** implied volatility surface:

$$w(k,t) = \frac{\theta(t)}{2} \left(1 + \rho \phi k + \sqrt{(\phi k + \rho)^2 + (1-\rho^2)}\right)$$

**Parameters:**
| Symbol | Name | Description | Range |
|--------|------|-------------|-------|
| $\eta$ | eta | Volatility of volatility | (0, ∞) |
| $\lambda$ | lam | Power law decay | (-1, 1) |
| $\rho$ | rho | Spot-vol correlation | (-1, 1) |
| $v_0$ | v0 | ATM variance rate | (0, ∞) |

### 1.2 Schrödinger Bridge Problem

The Schrödinger bridge finds the most likely path distribution:

$$\min_P \text{KL}(P \| R) \quad \text{subject to} \quad P_0 = \mu_0, \quad P_T = \mu_T$$

**Connection to HJB Equation:**
\frac{\partial V}{\partial t} = \min_{\mathbf{u}} \left\lbrace L(\mathbf{x}, \mathbf{u}, t) + \nabla V \cdot \mathbf{f}(\mathbf{x}, \mathbf{u}, t) + \frac{1}{2}\text{tr}\left(GG^T \nabla^2 V\right) \right\rbrace

### 1.3 Gibbs Field Interpretation

Path distributions follow Gibbs measure:

$$P(\omega) \propto \exp(-\beta H[\omega])$$

**Triple Equivalence Theorem:**
1. **Gibbs Field**: $P(\omega) \propto \exp(-\beta \sum_t L(x_t, x_{t+1}))$
2. **Stochastic Control**: $\min \mathbb{E}[\sum \|x_t - \hat{y}_t\|^2]$
3. **Path Integral**: $p(x_{t+1}|x_{0:t}) = \int \mathcal{D}[\omega] \exp(-S_E[\omega])$

---

## 2. Implementation Details

### 2.1 Package Structure: `schrodinger_finance/`

```
core/
├── graph_sobolev.py    # Discrete Sobolev spaces, Laplacian operators
├── nlse_solver.py      # NLSE with dissipation: dψ/dt = -iHψ - γPψ⊥D
├── landau_lifshitz.py  # Landau-Lifshitz-Gilbert dynamics
└── moduli_space.py     # Moduli space of gauge connections

optimal_transport/
├── sinkhorn.py         # Entropic regularization: K = exp(-C/ε)
├── schrodinger_bridge.py # Bridge between marginal distributions
└── hjb_solver.py       # HJB equation solver

finance/
├── heston_model.py     # Heston: dV = κ(θ-V)dt + σ√V dW
├── option_pricing.py   # Black-Scholes analytical formulas
├── calibration.py      # Parameter estimation framework
└── volatility_surface.py # Surface interpolation
```

---

## 3. Plots

### 3.1 SSVI Surface Plots

| File | Description |
|------|-------------|
| `ssvi_surface.png` | 3D implied volatility surface with contour map |
| `volatility_smiles.png` | Volatility smiles at T = 0.25, 0.5, 1.0, 2.0 years |
| `calibration_results.png` | Market vs model IV comparison with scatter plot |
| `term_structure.png` | IV term structure for multiple strikes |

### 3.2 OT Calibration Plots

| File | Description |
|------|-------------|
| `ot_ssvi_surface.png` | 3D surface with market points overlay |
| `ot_calibration_fit.png` | 4-panel: scatter, smiles, term structure, error histogram |
| `ot_option_prices.png` | 4-panel: prices, IV smile, delta, put-call parity |

### 3.3 Core Module Demo Plots

| File | Description |
|------|-------------|
| `demo_graph_sobolev.png` | Graph Laplacian eigenfunctions |
| `demo_ground_state.png` | NLSE ground state energy minimization |
| `demo_optimal_transport.png` | OT plan visualization |
| `demo_heston.png` | Heston Monte Carlo paths |
| `demo_heston_model.png` | Heston calibration fit |
| `demo_calibration.png` | Calibration convergence |
| `demo_summary.png` | Summary dashboard |

---

## 4. Usage Examples

### 4.1 Basic SSVI Calibration

```python
from ssvi_calibration import SSVICalibrator, ssvi_implied_vol

# Prepare market data
market_data = {
    'strikes': np.array([80, 90, 100, 110, 120]),
    'maturities': np.array([0.25, 0.5, 1.0]),
    'implied_vols': market_iv_surface,  # 2D array
    'spot': 100.0,
    'rate': 0.05
}

# Calibrate
calibrator = SSVICalibrator(market_data)
params = calibrator.calibrate_with_ot()

# Use calibrated model
iv = ssvi_implied_vol(k=0.1, t=0.5, **params)
```

### 4.2 Enhanced OT Calibration

```python
from ot_calibration import SchrodingerBridgeCalibrator

calibrator = SchrodingerBridgeCalibrator(market_data)
params = calibrator.calibrate(use_global=True)

# Check arbitrage
arb_check = calibrator.check_arbitrage()
print(f"Butterfly: {arb_check['butterfly']}")
print(f"Calendar: {arb_check['calendar']}")

# Get option prices
prices, ivs = calibrator.compute_option_prices(
    strikes=[90, 100, 110], 
    maturity=0.25
)
```

### 4.3 Using the Package

```python
from schrodinger_finance import (
    GraphSobolevSpace,
    NLSESolver,
    SinkhornSolver,
    HestonModel,
    BlackScholes
)

# Graph Sobolev space
graph = GraphSobolevSpace(adjacency_matrix)
laplacian = graph.discrete_laplacian()

# NLSE solver
nlse = NLSESolver(graph)
ground_state = nlse.compute_ground_state()

# OT solver
ot = SinkhornSolver(cost_matrix, epsilon=0.1)
plan = ot.solve(mu_0, mu_T)
```
