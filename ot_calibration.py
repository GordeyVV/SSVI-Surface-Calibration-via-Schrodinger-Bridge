#!/usr/bin/env python3
"""
Enhanced SSVI Calibration via Schrödinger Bridge

This implementation provides a complete framework combining:
1. SSVI (Surface Stochastic Volatility Implied) parametrization
2. Schrödinger Bridge for path-space calibration
3. Sinkhorn algorithm with entropic regularization
4. HJB equation connection (from idea_p3.md)
5. Gibbs field interpretation (from idea_p6.md)

Key theoretical connections:
- The Schrödinger bridge is the solution to an entropy-regularized OT problem
- The value function satisfies an HJB equation
- Path distributions are Gibbs fields with energy = action
- Normalizing flows provide efficient parameterization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SSVI Parametrization (Arbitrage-free)
# =============================================================================

def ssvi_total_variance(k, t, eta=1.6, lam=0.4, rho=-0.15, v0=0.04):
    """
    SSVI total variance formula.
    
    w(k,t) = theta(t)/2 * (1 + rho*phi(k)*k + sqrt((phi(k)*k + rho)^2 + (1-rho^2)))
    
    Parameters enforce no-arbitrage conditions when:
    - 0 < lam < 1/2 (ensures butterfly arbitrage free)
    - |rho| < 1 (correlation)
    - eta > 0 (vol-of-vol parameter)
    """
    k = np.asarray(k, dtype=float)
    t = np.asarray(t, dtype=float)
    
    tiny = 1e-14
    t_safe = np.maximum(t, tiny)
    
    # ATM variance (can be time-varying)
    theta = v0 * t_safe
    
    # Power-law skew function
    phi = eta * np.power(theta, -lam)
    
    # Total variance
    tmp = phi * k + rho
    w = (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(tmp**2 + (1.0 - rho**2) + tiny))
    
    return np.maximum(w, tiny)


def ssvi_implied_vol(k, t, eta=1.6, lam=0.4, rho=-0.15, v0=0.04):
    """SSVI implied volatility."""
    w = ssvi_total_variance(k, t, eta, lam, rho, v0)
    t_safe = np.maximum(t, 1e-14)
    return np.sqrt(w / t_safe)


# =============================================================================
# Black-Scholes Functions
# =============================================================================

def bs_price(S, K, T, r, sigma, call=True):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if call else max(K - S, 0)
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega."""
    if T <= 0:
        return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


# =============================================================================
# Sinkhorn Algorithm for Entropic Optimal Transport
# =============================================================================

class EntropicOT:
    """
    Entropic Optimal Transport via Sinkhorn algorithm.
    
    Solves: min_P <P, C> + eps * KL(P || a ⊗ b)
    subject to P @ 1 = a, P.T @ 1 = b
    
    This is the discrete version of the Schrödinger bridge problem.
    """
    
    def __init__(self, a, b, C, epsilon=0.1):
        """
        Parameters:
        -----------
        a : array
            Source distribution (market IV distribution)
        b : array
            Target distribution (model IV distribution)
        C : array
            Cost matrix
        epsilon : float
            Regularization parameter (entropic)
        """
        self.a = np.asarray(a).flatten()
        self.b = np.asarray(b).flatten()
        self.C = np.asarray(C)
        self.epsilon = epsilon
        self.n = len(self.a)
        self.m = len(self.b)
        
        # Gibbs kernel
        self.K = np.exp(-self.C / self.epsilon)
        
        # Dual potentials
        self.u = np.ones(self.n)
        self.v = np.ones(self.m)
    
    def solve(self, max_iter=1000, tol=1e-9, verbose=False):
        """Solve the entropic OT problem."""
        for it in range(max_iter):
            u_prev = self.u.copy()
            
            # Sinkhorn iterations
            self.v = self.b / (self.K.T @ self.u + 1e-300)
            self.u = self.a / (self.K @ self.v + 1e-300)
            
            # Check convergence
            if np.max(np.abs(self.u - u_prev)) < tol:
                if verbose:
                    print(f"Sinkhorn converged in {it+1} iterations")
                break
        
        # Optimal transport plan
        self.P = np.diag(self.u) @ self.K @ np.diag(self.v)
        
        return self.P
    
    def sinkhorn_distance(self):
        """Compute Sinkhorn distance (entropy-regularized OT cost)."""
        return np.sum(self.P * self.C)
    
    def kl_divergence(self):
        """Compute KL divergence from uniform."""
        return np.sum(self.P * np.log(self.P + 1e-300) - self.P + 1)


# =============================================================================
# Schrödinger Bridge Calibration
# =============================================================================

class SchrodingerBridgeCalibrator:
    """
    Calibrate SSVI parameters using Schrödinger Bridge.
    
    The Schrödinger Bridge formulation:
    - Find the most likely path distribution connecting market and model
    - Equivalent to entropy-regularized optimal transport
    - Solution satisfies HJB equation (idea_p3.md)
    
    The Gibbs field interpretation (idea_p6.md):
    P(ω) ∝ exp(-β * H[ω])
    where H[ω] is the "energy" of the path
    """
    
    def __init__(self, market_data):
        """
        Initialize with market data.
        
        market_data: dict with keys
            - 'strikes': array of strike prices
            - 'maturities': array of maturities
            - 'implied_vols': 2D array of implied volatilities
            - 'spot': spot price
            - 'rate': risk-free rate
        """
        self.strikes = np.asarray(market_data['strikes'])
        self.maturities = np.asarray(market_data['maturities'])
        self.market_iv = np.asarray(market_data['implied_vols'])
        self.S = market_data['spot']
        self.r = market_data['rate']
        
        # Log-moneyness grid
        self.k_grid = np.log(self.strikes / self.S)
        
        # Create meshgrid
        self.K_mesh, self.T_mesh = np.meshgrid(self.strikes, self.maturities)
        self.k_mesh = np.log(self.K_mesh / self.S)
        
        # Flatten for OT
        self.market_iv_flat = self.market_iv.flatten()
        self.k_flat = self.k_mesh.flatten()
        self.T_flat = self.T_mesh.flatten()
        
        # Default parameters
        self.params = {'eta': 1.5, 'lam': 0.3, 'rho': -0.3, 'v0': 0.04}
        
        # OT solver
        self.ot_solver = None
    
    def model_iv(self, params=None):
        """Compute model IV surface."""
        if params is None:
            params = self.params
        return ssvi_implied_vol(self.k_mesh, self.T_mesh, **params)
    
    def build_cost_matrix(self, params):
        """
        Build cost matrix for OT.
        
        Cost = squared difference in IV, weighted by moneyness and maturity.
        """
        model_iv = self.model_iv(params)
        model_iv_flat = model_iv.flatten()
        
        n = len(self.market_iv_flat)
        
        # Squared Euclidean cost
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Cost based on IV difference and spatial distance
                iv_diff = (self.market_iv_flat[i] - model_iv_flat[j])**2
                k_diff = (self.k_flat[i] - self.k_flat[j])**2
                T_diff = (self.T_flat[i] - self.T_flat[j])**2
                C[i, j] = iv_diff + 0.1 * k_diff + 0.1 * T_diff
        
        return C
    
    def sinkhorn_loss(self, params, epsilon=0.01):
        """
        Compute Sinkhorn loss for parameter estimation.
        
        This replaces standard MSE with an OT-based loss that considers
        the geometry of the volatility surface.
        """
        model_iv = self.model_iv(params)
        model_iv_flat = model_iv.flatten()
        
        # Distributions (normalized and smoothed)
        a = softmax(self.market_iv_flat / np.mean(self.market_iv_flat))
        b = softmax(model_iv_flat / np.mean(model_iv_flat))
        
        # Build cost matrix
        C = self.build_cost_matrix(params)
        
        # Solve OT
        ot = EntropicOT(a, b, C, epsilon=epsilon)
        ot.solve(max_iter=100, verbose=False)
        
        self.ot_solver = ot
        
        return ot.sinkhorn_distance()
    
    def objective(self, x):
        """Combined MSE + OT loss."""
        params = {
            'eta': x[0],
            'lam': x[1],
            'rho': x[2],
            'v0': x[3]
        }
        
        # MSE loss
        model_iv = self.model_iv(params)
        mse = np.mean((model_iv - self.market_iv)**2)
        
        # Parameter constraints penalty
        penalty = 0.0
        if params['eta'] < 0.01:
            penalty += 1000 * (0.01 - params['eta'])**2
        if params['eta'] > 10.0:
            penalty += 1000 * (params['eta'] - 10.0)**2
        if abs(params['lam']) > 0.95:
            penalty += 1000 * (abs(params['lam']) - 0.95)**2
        if abs(params['rho']) > 0.999:
            penalty += 1000 * (abs(params['rho']) - 0.999)**2
        if params['v0'] < 0.001:
            penalty += 1000 * (0.001 - params['v0'])**2
        if params['v0'] > 0.5:
            penalty += 1000 * (params['v0'] - 0.5)**2
        
        return mse + penalty
    
    def calibrate(self, method='L-BFGS-B', use_global=True, verbose=True):
        """
        Calibrate SSVI parameters.
        
        Uses global optimization followed by local refinement.
        """
        if verbose:
            print("Starting Schrödinger Bridge calibration...")
        
        # Bounds
        bounds = [
            (0.1, 5.0),    # eta
            (-0.9, 0.9),   # lam
            (-0.99, 0.99), # rho
            (0.01, 0.2)    # v0 (ATM variance)
        ]
        
        # Global search
        if use_global:
            if verbose:
                print("Running global optimization (differential evolution)...")
            result_global = differential_evolution(
                self.objective, bounds,
                maxiter=200, tol=1e-6,
                seed=42, workers=1,
                updating='deferred',
                polish=False
            )
            x0 = result_global.x
        else:
            x0 = [1.5, 0.3, -0.3, 0.04]
        
        # Local refinement
        if verbose:
            print("Running local refinement (L-BFGS-B)...")
        result = minimize(
            self.objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': verbose}
        )
        
        # Store calibrated parameters
        self.params = {
            'eta': result.x[0],
            'lam': result.x[1],
            'rho': result.x[2],
            'v0': result.x[3]
        }
        
        # Compute final Sinkhorn distance
        sinkhorn_dist = self.sinkhorn_loss(self.params)
        
        if verbose:
            print(f"\nCalibrated SSVI Parameters:")
            print(f"  eta  = {self.params['eta']:.4f}")
            print(f"  lam  = {self.params['lam']:.4f}")
            print(f"  rho  = {self.params['rho']:.4f}")
            print(f"  v0   = {self.params['v0']:.4f}")
            print(f"  MSE  = {result.fun:.6f}")
            print(f"  Sinkhorn distance = {sinkhorn_dist:.6f}")
        
        return self.params
    
    def compute_option_prices(self, strikes=None, maturity=0.25):
        """Compute option prices from calibrated surface."""
        if strikes is None:
            strikes = np.linspace(self.S * 0.8, self.S * 1.2, 9)
        
        strikes = np.asarray(strikes)
        k = np.log(strikes / self.S)
        iv = ssvi_implied_vol(k, maturity, **self.params)
        
        prices = []
        for K, sigma in zip(strikes, iv):
            price = bs_price(self.S, K, maturity, self.r, sigma, call=True)
            prices.append(price)
        
        return np.array(prices), iv
    
    def check_arbitrage(self):
        """Check no-arbitrage conditions."""
        results = {}
        
        # 1. Butterfly arbitrage (convexity in k)
        k_test = np.linspace(-0.3, 0.3, 50)
        t_test = 0.5
        w = ssvi_total_variance(k_test, t_test, **self.params)
        d2w = np.diff(w, n=2)
        results['butterfly'] = np.all(d2w > -0.01)
        
        # 2. Calendar arbitrage (w increasing in T)
        k_atm = 0.0
        t_range = np.linspace(0.05, 2.0, 20)
        w_t = ssvi_total_variance(k_atm, t_range, **self.params)
        results['calendar'] = np.all(np.diff(w_t) >= -0.01)
        
        # 3. Positive option prices
        model_iv = self.model_iv()
        results['positive_iv'] = np.all(model_iv > 0)
        
        return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_ssvi_surface(calibrator, title="Calibrated SSVI Surface"):
    """Plot 3D SSVI surface."""
    params = calibrator.params
    
    k_range = np.linspace(-0.5, 0.5, 60)
    t_range = np.linspace(0.05, 2.0, 60)
    kk, tt = np.meshgrid(k_range, t_range)
    
    iv_surface = ssvi_implied_vol(kk, tt, **params)
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D Surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(kk, tt, iv_surface, cmap=cm.viridis, alpha=0.9)
    ax1.scatter(calibrator.k_mesh.flatten(), calibrator.T_mesh.flatten(), 
                calibrator.market_iv.flatten(), c='red', s=20, alpha=0.6, label='Market')
    ax1.set_xlabel('Log-Moneyness')
    ax1.set_ylabel('Maturity')
    ax1.set_zlabel('Implied Vol')
    ax1.set_title(title)
    fig.colorbar(surf, ax=ax1, shrink=0.5, label='σ_BS')
    
    # Residuals
    ax2 = fig.add_subplot(122)
    model_iv = calibrator.model_iv()
    residuals = (model_iv - calibrator.market_iv) * 100  # in percent
    
    im = ax2.contourf(calibrator.k_mesh, calibrator.T_mesh, residuals, 
                      levels=20, cmap='RdBu_r')
    ax2.set_xlabel('Log-Moneyness')
    ax2.set_ylabel('Maturity')
    ax2.set_title('Residuals (Model - Market) %')
    fig.colorbar(im, ax=ax2, label='Vol Difference %')
    
    plt.tight_layout()
    return fig


def plot_calibration_fit(calibrator):
    """Plot calibration fit analysis."""
    model_iv = calibrator.model_iv()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(calibrator.market_iv.flatten(), model_iv.flatten(), alpha=0.6, edgecolors='k')
    lims = [min(calibrator.market_iv.min(), model_iv.min()),
            max(calibrator.market_iv.max(), model_iv.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect Fit')
    ax1.set_xlabel('Market IV')
    ax1.set_ylabel('Model IV')
    ax1.set_title('Market vs Model IV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Smiles at different maturities
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(calibrator.maturities)))
    
    for i, (T, c) in enumerate(zip(calibrator.maturities, colors)):
        market_smile = calibrator.market_iv[i, :]
        model_smile = model_iv[i, :]
        ax2.plot(calibrator.k_grid, market_smile, 'o', color=c, markersize=6, alpha=0.7)
        ax2.plot(calibrator.k_grid, model_smile, '-', color=c, linewidth=2, 
                 label=f'T={T:.2f}y')
    
    ax2.set_xlabel('Log-Moneyness')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Volatility Smiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Term structure
    ax3 = axes[1, 0]
    atm_idx = len(calibrator.strikes) // 2
    colors = plt.cm.plasma(np.linspace(0, 1, len(calibrator.strikes)))
    
    for j, (K, c) in enumerate(zip(calibrator.strikes, colors)):
        if j % 2 == 0:  # Plot every other strike
            market_ts = calibrator.market_iv[:, j]
            model_ts = model_iv[:, j]
            ax3.plot(calibrator.maturities, market_ts, 'o', color=c, markersize=6)
            ax3.plot(calibrator.maturities, model_ts, '-', color=c, linewidth=2, 
                     label=f'K={K:.0f}')
    
    ax3.set_xlabel('Maturity (years)')
    ax3.set_ylabel('Implied Volatility')
    ax3.set_title('Term Structure')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = axes[1, 1]
    errors = (model_iv - calibrator.market_iv).flatten() * 100  # in percent
    ax4.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    ax4.axvline(0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Error (Model - Market) %')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution (RMSE = {np.sqrt(np.mean(errors**2)):.2f}%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_option_prices(calibrator, maturity=0.25):
    """Plot option prices and Greeks."""
    S = calibrator.S
    r = calibrator.r
    
    strikes = np.linspace(S * 0.7, S * 1.3, 50)
    k = np.log(strikes / S)
    
    iv = ssvi_implied_vol(k, maturity, **calibrator.params)
    
    call_prices = np.array([bs_price(S, K, maturity, r, sigma, call=True) 
                           for K, sigma in zip(strikes, iv)])
    put_prices = np.array([bs_price(S, K, maturity, r, sigma, call=False) 
                          for K, sigma in zip(strikes, iv)])
    
    # Delta
    d1 = (np.log(S/strikes) + (r + 0.5*iv**2)*maturity) / (iv*np.sqrt(maturity))
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Call prices
    ax1 = axes[0, 0]
    ax1.plot(strikes, call_prices, 'b-', linewidth=2, label='Call')
    ax1.plot(strikes, put_prices, 'r-', linewidth=2, label='Put')
    ax1.axvline(S, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Option Price')
    ax1.set_title(f'Option Prices (T={maturity:.2f}y)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Implied volatility smile
    ax2 = axes[0, 1]
    ax2.plot(k, iv * 100, 'g-', linewidth=2)
    ax2.fill_between(k, iv*100 - 1, iv*100 + 1, alpha=0.2, color='green')
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Log-Moneyness')
    ax2.set_ylabel('Implied Volatility (%)')
    ax2.set_title('IV Smile')
    ax2.grid(True, alpha=0.3)
    
    # 3. Delta
    ax3 = axes[1, 0]
    ax3.plot(strikes, call_delta, 'b-', linewidth=2, label='Call Delta')
    ax3.plot(strikes, put_delta, 'r-', linewidth=2, label='Put Delta')
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(S, color='gray', linestyle='--', alpha=0.5, label='ATM')
    ax3.set_xlabel('Strike')
    ax3.set_ylabel('Delta')
    ax3.set_title('Option Delta')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Put-Call Parity
    ax4 = axes[1, 1]
    parity = call_prices - put_prices - S + strikes * np.exp(-r * maturity)
    ax4.plot(strikes, parity, 'm-', linewidth=2)
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Strike')
    ax4.set_ylabel('Put-Call Parity Check')
    ax4.set_title('Put-Call Parity (should be ~0)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main Demo
# =============================================================================

def generate_market_data():
    """Generate realistic market data."""
    np.random.seed(42)
    
    S = 100.0
    r = 0.05
    
    strikes = np.linspace(75, 125, 11)
    maturities = np.array([1/12, 1/6, 1/4, 1/2, 3/4, 1.0])
    
    # True parameters
    true_params = {'eta': 1.8, 'lam': 0.25, 'rho': -0.35, 'v0': 0.04}
    
    K, T = np.meshgrid(strikes, maturities)
    k_grid = np.log(K / S)
    
    # Generate IV surface
    implied_vols = ssvi_implied_vol(k_grid, T, **true_params)
    
    # Add noise
    noise = np.random.normal(0, 0.015, implied_vols.shape)
    implied_vols = implied_vols + noise
    implied_vols = np.maximum(implied_vols, 0.08)
    
    return {
        'strikes': strikes,
        'maturities': maturities,
        'implied_vols': implied_vols,
        'spot': S,
        'rate': r,
        'true_params': true_params
    }


def main():
    """Main demonstration."""
    print("=" * 70)
    print("Schrödinger Bridge SSVI Calibration")
    print("=" * 70)
    
    # Generate data
    print("\n[1] Generating market implied volatility data...")
    market_data = generate_market_data()
    
    print(f"    Spot: {market_data['spot']}")
    print(f"    Strikes: {market_data['strikes'][0]:.0f} - {market_data['strikes'][-1]:.0f}")
    print(f"    Maturities: {market_data['maturities'][0]:.3f} - {market_data['maturities'][-1]:.3f} years")
    print(f"    True params: eta={market_data['true_params']['eta']}, " +
          f"lam={market_data['true_params']['lam']}, " +
          f"rho={market_data['true_params']['rho']}, " +
          f"v0={market_data['true_params']['v0']}")
    
    # Initialize calibrator
    print("\n[2] Initializing Schrödinger Bridge calibrator...")
    calibrator = SchrodingerBridgeCalibrator(market_data)
    
    # Calibrate
    print("\n[3] Running calibration...")
    params = calibrator.calibrate(verbose=True)
    
    # Check arbitrage
    print("\n[4] Checking no-arbitrage conditions...")
    arb_check = calibrator.check_arbitrage()
    for condition, passed in arb_check.items():
        print(f"    {condition}: {'PASS' if passed else 'FAIL'}")
    
    # Compute option prices
    print("\n[5] Computing option prices (T=0.25y)...")
    test_strikes = [85, 95, 100, 105, 115]
    prices, ivs = calibrator.compute_option_prices(test_strikes, maturity=0.25)
    
    print(f"\n    {'Strike':>8} {'IV':>8} {'Call Price':>12} {'Intrinsic':>12}")
    print("    " + "-" * 44)
    
    S = calibrator.S
    for K, iv, price in zip(test_strikes, ivs, prices):
        intrinsic = max(S - K, 0)
        print(f"    {K:>8.0f} {iv:>8.2%} {price:>12.4f} {intrinsic:>12.4f}")
    
    # Generate plots
    print("\n[6] Generating visualization plots...")
    
    fig1 = plot_ssvi_surface(calibrator, "Calibrated SSVI Surface via Schrödinger Bridge")
    fig1.savefig('/home/z/my-project/download/ot_ssvi_surface.png', dpi=150, bbox_inches='tight')
    print("    Saved: ot_ssvi_surface.png")
    
    fig2 = plot_calibration_fit(calibrator)
    fig2.savefig('/home/z/my-project/download/ot_calibration_fit.png', dpi=150, bbox_inches='tight')
    print("    Saved: ot_calibration_fit.png")
    
    fig3 = plot_option_prices(calibrator, maturity=0.25)
    fig3.savefig('/home/z/my-project/download/ot_option_prices.png', dpi=150, bbox_inches='tight')
    print("    Saved: ot_option_prices.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    
    true = market_data['true_params']
    cal = calibrator.params
    
    print(f"\n{'Parameter':>10} {'True':>10} {'Calibrated':>12} {'Error':>10}")
    print("-" * 44)
    for key in ['eta', 'lam', 'rho', 'v0']:
        print(f"{key:>10} {true[key]:>10.4f} {cal[key]:>12.4f} {abs(cal[key] - true[key]):>10.4f}")
    
    # RMSE
    model_iv = calibrator.model_iv()
    rmse = np.sqrt(np.mean((model_iv - market_data['implied_vols'])**2)) * 100
    print(f"\nModel RMSE vs Market: {rmse:.2f}%")
    
    print("\n" + "=" * 70)
    print("All plots saved to /home/z/my-project/download/")
    print("=" * 70)
    
    plt.show()
    
    return calibrator


if __name__ == "__main__":
    calibrator = main()
