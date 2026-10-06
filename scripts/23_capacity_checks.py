#!/usr/bin/env python3
"""23 - Capacity in the two-factor equilibrium: verification of Section 12.

(a) E[d lam | lam = 0] = kappa_0 lbar dt with kappa_0 = (sigma_a^2 + b^2 sigma0^2)/(2 V_lam): Gaussian conditioning
    identity on random parameters, and a Monte Carlo check of the conditional drift near lam = 0.
(b) Recovery after a crossing: E[lam_{t+tau} - lbar | z_t] = l' e^{M tau} z_t, against the closed forms for a
    supply-driven crossing (deficit lbar e^{-kappa tau}) and a demand-driven one (deficit
    lbar [e^{-ka tau} - b g (e^{-ka tau} - e^{-kappa tau})/(kappa - ka)]), and the demand-driven deficit is below
    the demand-only deficit lbar e^{-ka tau} for every tau > 0.
(c) P(lam <= 0) = Phi(-lbar/sqrt(V_lam)) is increasing in sigma_a.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402

rng = np.random.default_rng(23)
dev_a, dev_b, below, mono = 0.0, 0.0, 0, 0
for _ in range(300):
    eta, rho, chi, b, ka = 10 ** rng.uniform(-0.7, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3), \
        10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3)
    s0, sa, abar = 10 ** rng.uniform(-1.5, 0), 10 ** rng.uniform(-1.5, 0.5), 1.0
    cf = TF.closed_forms(eta, rho, chi, b, ka, abar)
    k, g, lbar = cf["kappa"], cf["g"], cf["lbar"]
    M = np.array([[-k, g], [0.0, -ka]])
    S = solve_continuous_lyapunov(M, -np.diag([s0 ** 2, sa ** 2]))
    l = np.array([-b, 1.0])
    V = l @ S @ l
    drift_coef = (l @ M @ S @ l) / V                      # E[d lam | lam - lbar = w] = drift_coef * w
    kappa0 = (sa ** 2 + b ** 2 * s0 ** 2) / (2 * V)
    dev_a = max(dev_a, abs(-drift_coef - kappa0) / kappa0)
    taus = np.linspace(0.0, 10 / min(k, ka), 400)[1:]
    for z0, closed in ((np.array([lbar / b, 0.0]), lambda t: lbar * np.exp(-k * t)),
                       (np.array([0.0, -lbar]), lambda t: lbar * (np.exp(-ka * t) - b * g * (np.exp(-ka * t) - np.exp(-k * t)) / (k - ka)))):
        for t in taus[::40]:
            deficit = -(l @ expm(M * t) @ z0)
            dev_b = max(dev_b, abs(deficit - closed(t)) / lbar)
    dem = lbar * (np.exp(-ka * taus) - b * g * (np.exp(-ka * taus) - np.exp(-k * taus)) / (k - ka))
    below += int(np.all(dem < lbar * np.exp(-ka * taus) + 1e-15))
    P = [norm.logcdf(-lbar / np.sqrt(np.array([-b, 1.0]) @ solve_continuous_lyapunov(M, -np.diag([s0 ** 2, x ** 2])) @ np.array([-b, 1.0])))
         for x in (0.5 * sa, sa, 2 * sa)]
    mono += int(P[0] < P[1] < P[2])
print(f"(a) kappa_0 identity: max rel. deviation {dev_a:.1e} (300 sets)")
print(f"(b) recovery paths vs closed forms: max deviation {dev_b:.1e}; demand-driven deficit below demand-only deficit: {below} of 300")
print(f"(c) P(lam <= 0) increasing in sigma_a (computed on the log scale): {mono} of 300")

# Monte Carlo of the conditional drift at lam = 0 (one parameter set)
cf = TF.closed_forms(1.0, 0.10, 0.04, 0.05, 2.0, 0.05)
k, g, lbar = cf["kappa"], cf["g"], cf["lbar"]
b, s0, sa = 0.05, 0.15, 0.02
M = np.array([[-k, g], [0.0, -2.0]])
S = solve_continuous_lyapunov(M, -np.diag([s0 ** 2, sa ** 2]))
l = np.array([-b, 1.0]); V = l @ S @ l
kappa0 = (sa ** 2 + b ** 2 * s0 ** 2) / (2 * V)
dt = 0.01
A = expm(M * dt)
L = np.linalg.cholesky(S - A @ S @ A.T)
n = 2_000_000
z = np.zeros((n, 2)); z[0] = np.linalg.cholesky(S) @ rng.standard_normal(2)
eps = rng.standard_normal((n, 2)) @ L.T
for t in range(1, n):
    z[t] = A @ z[t - 1] + eps[t]
lam = lbar + z @ l
near = np.abs(lam[:-1]) < 0.05 * np.sqrt(V)
d = (lam[1:] - lam[:-1])[near] / dt
print(f"MC: conditional drift at lam = 0: {d.mean():.5f} (s.e. {d.std(ddof=1) / np.sqrt(near.sum()):.5f}) vs kappa_0 lbar = {kappa0 * lbar:.5f}; "
      f"kappa_0 = {kappa0:.3f} (kappa = {k:.3f}, kappa_a = 2.0); P(lam <= 0) = {norm.cdf(-lbar / np.sqrt(V)):.3f}")

# (d) decomposition of the restoring drift at the manifold into the supply regulator and demand reversion
dev_d, pos_s, pos_d, neg_state = 0.0, 0, 0, 0
for _ in range(300):
    eta, rho, chi, b, ka = 10 ** rng.uniform(-0.7, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3), \
        10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3)
    s0, sa = 10 ** rng.uniform(-1.5, 0), 10 ** rng.uniform(-1.5, 0.5)
    cf = TF.closed_forms(eta, rho, chi, b, ka, 1.0)
    k, g, lbar = cf["kappa"], cf["g"], cf["lbar"]
    M = np.array([[-k, g], [0.0, -ka]])
    S = solve_continuous_lyapunov(M, -np.diag([s0 ** 2, sa ** 2]))
    l = np.array([-b, 1.0]); V = l @ S @ l
    Ez = S @ l * (-lbar) / V                               # E[(y, alpha) | lam = 0]
    supply = -b * (-k * Ez[0] + g * Ez[1])                 # -b x (expected drift of y)
    demand = -ka * Ez[1]                                   # expected drift of alpha
    sup_cf = lbar * (b ** 2 * s0 ** 2 + b * g * sa ** 2 / (k + ka)) / (2 * V)
    dem_cf = lbar * sa ** 2 * (1 - b * g / (k + ka)) / (2 * V)
    dev_d = max(dev_d, abs(supply - sup_cf) / abs(sup_cf), abs(demand - dem_cf) / abs(dem_cf))
    pos_s += int(supply > 0); pos_d += int(demand > 0)
    alpha = -10 * lbar                                     # a deep demand collapse on the manifold
    y = (alpha + lbar) / b
    neg_state += int(-b * (-k * y + g * alpha) < 0)
print(f"(d) restoring drift decomposition vs closed forms: max rel. deviation {dev_d:.1e}; supply part > 0: {pos_s}/300, "
      f"demand part > 0: {pos_d}/300; states on the manifold where the supply regulator deepens the crossing "
      f"(alpha = -10 lbar): {neg_state}/300")
