"""Numerical tools: stationary densities of scalar diffusions, moments, and a
monotone (upwind) policy-iteration solver for the auxiliary control problem
of Theorem 4.1 with a general clearing curve.

Symbols
  y          grid of aggregate states Y
  drift      aggregate drift b_hat(Y) evaluated on the grid
  dens       stationary density  rho_st(Y) proportional to exp((2/sigma0^2) * int b_hat)
             (written rho_st, not the paper's varrho, to avoid confusion with
             the discount rate rho)
  S          shadow potential; u = S' shadow price
  v          control of the auxiliary problem; its total drift is v - kappa_ind Y
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import cumulative_trapezoid, trapezoid

from . import model as M


# --------------------------------------------------------------------------
# stationary laws
# --------------------------------------------------------------------------
def stationary_density(drift_values: np.ndarray, y: np.ndarray, sigma0: float) -> np.ndarray:
    """Density of dY = b_hat(Y) dt + sigma0 dW on the grid (zero-flux solution)."""
    phi = (2.0 / sigma0 ** 2) * cumulative_trapezoid(drift_values, y, initial=0.0)
    phi -= phi.max()
    dens = np.exp(phi)
    return dens / trapezoid(dens, y)


def moments(y: np.ndarray, dens: np.ndarray, values: np.ndarray | None = None) -> tuple[float, float, float]:
    """(mean, variance, skewness) of values(Y) under dens (values default to Y)."""
    x = y if values is None else values
    m = trapezoid(x * dens, y)
    d = x - m
    v = trapezoid(d * d * dens, y)
    m3 = trapezoid(d ** 3 * dens, y)
    return float(m), float(v), float(m3 / v ** 1.5)


def table2_ratio(c: float, kap: float, eta: float = 1.0, sigma0: float = 0.15,
                 rho: float = 0.10, ybar: float = 0.5, width: float = 6.0, n: int = 60001) -> float:
    """The paper's Table 2 procedure: numerical skewness of Y under the
    first-order drift on Ybar +/- width*s_star, divided by eq. (23)."""
    s_st = sigma0 / np.sqrt(2.0 * kap)
    y = np.linspace(ybar - width * s_st, ybar + width * s_st, n)
    drift = M.first_order_drift(rho, sigma0, eta, kap, ybar, c)(y)
    dens = stationary_density(drift, y, sigma0)
    return moments(y, dens)[2] / M.skew_Y_eq23(rho, sigma0, eta, kap, c)


def skewness_first_order_drift(p: M.Params, c: float, width: float = 6.0, n: int = 60001) -> dict:
    """Numerical skewness of Y and of lam = Lambda_c(Y) under the paper's own
    first-order drift, on the operating region Ybar +/- width*s_star."""
    kap, yb, s_st = M.kappa(p), M.Ybar(p), M.s_star(p)
    y = np.linspace(yb - width * s_st, yb + width * s_st, n)
    drift = M.first_order_drift(p.rho, p.sigma0, p.eta, kap, yb, c)(y)
    dens = stationary_density(drift, y, p.sigma0)
    lam = p.a - p.b * y - 0.5 * c * y * y
    return dict(skew_Y=moments(y, dens)[2], skew_lam=moments(y, dens, lam)[2])


# --------------------------------------------------------------------------
# nonlinear auxiliary problem (Theorem 4.1 / Proposition 8.2)
# --------------------------------------------------------------------------
def solve_shadow_potential(p: M.Params, lam_fn, lo: float, hi: float, n: int = 8001,
                           tol: float = 1e-9, maxit: int = 60) -> dict:
    """Solve   rho S = max_v [v S' - (eta/2) v^2] - kappa_ind Y S' + (sigma0^2/2) S'' + G(Y)
    on [lo, hi] with reflecting ends, where G' = lam_fn (any continuous clearing curve).

    Upwind finite differences + Howard policy iteration (monotone scheme).
    Returns the grid, S, the shadow price u = S' (second-order central
    differences after convergence), the aggregate drift b_hat = u/eta - kappa_ind Y,
    and the number of iterations.
    """
    y = np.linspace(lo, hi, n)
    h = y[1] - y[0]
    G = cumulative_trapezoid(lam_fn(y), y, initial=0.0)
    kin, eta, rho = M.kappa_ind(p), p.eta, p.rho
    dcoef = 0.5 * p.sigma0 ** 2 / h ** 2
    v = (M.A(p) * y + M.B(p)) / eta            # warm start: the affine solution
    ident = sp.identity(n, format="csc")
    S = np.zeros(n)
    it = 0
    for it in range(1, maxit + 1):
        mu = v - kin * y
        up = np.maximum(mu, 0.0) / h + dcoef
        dn = np.maximum(-mu, 0.0) / h + dcoef
        up[-1] = 0.0                           # reflecting upper end
        dn[0] = 0.0                            # reflecting lower end
        L = sp.diags([dn[1:], -(up + dn), up[:-1]], [-1, 0, 1], format="csc")
        S = spla.spsolve(rho * ident - L, G - 0.5 * eta * v * v)
        dF = np.zeros(n)
        dB = np.zeros(n)
        dF[:-1] = (S[1:] - S[:-1]) / h
        dB[1:] = (S[1:] - S[:-1]) / h
        vF, vB = dF / eta, dB / eta
        muF, muB = vF - kin * y, vB - kin * y
        HF = vF * dF - 0.5 * eta * vF ** 2 - kin * y * dF
        HB = vB * dB - 0.5 * eta * vB ** 2 - kin * y * dB
        useF, useB = muF > 0, muB < 0
        both = useF & useB
        v_new = np.where(useF & ~useB, vF,
                 np.where(useB & ~useF, vB,
                  np.where(both, np.where(HF >= HB, vF, vB), kin * y)))
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    u = np.gradient(S, h, edge_order=2)
    drift = u / eta - kin * y
    return dict(y=y, S=S, u=u, drift=drift, drift_upwind=v - kin * y, iterations=it)


def skewness_nonlinear(p: M.Params, c: float, R_mult: float = 6.0, grid_mult: float = 10.0,
                       n: int = 8001) -> dict:
    """Skewness of Y and lam = Lambda_c(Y) from the NON-perturbative equilibrium
    (policy iteration on the auxiliary problem with the truncated curved clearing)."""
    yb, s_st = M.Ybar(p), M.s_star(p)
    lam_fn = M.clearing_curved(p, c, R_mult * s_st)
    sol = solve_shadow_potential(p, lam_fn, yb - grid_mult * s_st, yb + grid_mult * s_st, n=n)
    dens = stationary_density(sol["drift"], sol["y"], p.sigma0)
    lam = lam_fn(sol["y"])
    mY, vY, skY = moments(sol["y"], dens)
    mL, vL, skL = moments(sol["y"], dens, lam)
    return dict(skew_Y=skY, skew_lam=skL, mean_Y=mY, var_Y=vY, mean_lam=mL, var_lam=vL,
                iterations=sol["iterations"], solution=sol)
