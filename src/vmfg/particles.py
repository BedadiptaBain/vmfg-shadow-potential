"""Monte Carlo checks.

simulate_particles: N sellers X^1..X^N driven by their own Brownian motions
W^1..W^N and one common W^0, each using the equilibrium feedback
v^i = (2 q0 X^i + A Ybar^N + B)/eta with Ybar^N the empirical mean.  For finite
N the empirical mean is an exact OU process with speed kappa (plus an extra
noise of size sigma/sqrt(N)); the deviations X^i - Ybar^N revert at kappa_ind.

simulate_aggregate: many independent copies of the reduced aggregate SDE
dY = b_hat(Y) dt + sigma0 dW0 with an arbitrary (e.g. numerically solved) drift.

Symbols: dt time step; T horizon; burn burn-in time discarded;
Yn empirical mean; disp cross-sectional variance (ddof = 0).
"""
from __future__ import annotations

import numpy as np

from . import model as M


def simulate_particles(p: M.Params, N: int = 300, T: float = 1500.0, dt: float = 0.02,
                       burn: float = 50.0, record_every: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    q, a_, b_ = M.q0(p), M.A(p), M.B(p)
    X = M.Ybar(p) + np.sqrt(max(M.Sigma_star(p), 1e-12)) * rng.standard_normal(N)
    nsteps = int(round(T / dt))
    every = max(1, int(round(record_every / dt)))
    sq = np.sqrt(dt)
    Ys, disp, Z_now, Z_next = [], [], [], []
    for k in range(nsteps):
        Yn = X.mean()
        v = (2.0 * q * X + a_ * Yn + b_) / p.eta
        dW0 = sq * rng.standard_normal()
        X_new = X + v * dt + p.sigma * sq * rng.standard_normal(N) + p.sigma0 * dW0
        if k * dt >= burn and k % every == 0:
            Ys.append(Yn)
            disp.append(X.var())
            if k % (every * 50) == 0:          # thinned pairs for the kappa_ind regression
                Z_now.append(X - Yn)
                Z_next.append(X_new - X_new.mean())
        X = X_new
    return dict(Y=np.array(Ys), disp=np.array(disp), dt_rec=every * dt,
                Z_now=np.concatenate(Z_now), Z_next=np.concatenate(Z_next), dt=dt, N=N)


def ar1_speed(y: np.ndarray, dt: float) -> tuple[float, float, float]:
    """Fit y_{k+1} = phi y_k + const + e; return (kappa_hat, mean, variance)."""
    x0, x1 = y[:-1], y[1:]
    xm = x0.mean()
    phi = np.sum((x0 - xm) * (x1 - x1.mean())) / np.sum((x0 - xm) ** 2)
    return -np.log(phi) / dt, float(y.mean()), float(y.var())


def simulate_aggregate(drift_fn, y0: float, sigma0: float, n_paths: int = 4000,
                       T: float = 200.0, dt: float = 0.02, burn: float = 30.0,
                       record_every: float = 0.5, seed: int = 1) -> np.ndarray:
    """Pooled samples of dY = drift_fn(Y) dt + sigma0 dW0 (n_paths independent copies).
    Returns an array (n_records, n_paths)."""
    rng = np.random.default_rng(seed)
    y = np.full(n_paths, y0, dtype=float)
    nsteps = int(round(T / dt))
    every = max(1, int(round(record_every / dt)))
    sq = np.sqrt(dt)
    out = []
    for k in range(nsteps):
        y = y + drift_fn(y) * dt + sigma0 * sq * rng.standard_normal(n_paths)
        if k * dt >= burn and k % every == 0:
            out.append(y.copy())
    return np.array(out)


def skewness(x: np.ndarray) -> float:
    d = x - x.mean()
    return float(np.mean(d ** 3) / np.mean(d ** 2) ** 1.5)
