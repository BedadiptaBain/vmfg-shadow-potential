"""What can the Section 10 statistics identify?

Synthetic NIFTY-like markets in which the TRUE ex-ante premium pi_t is known:
  H_unbounded : pi_t is an OU process on the whole line (Theorem 5.3's shape)
  H_reflected : pi_t is the same OU reflected at zero, so pi_t >= 0 always
In both, implied variance is IV2_t = E_t[RV_{t,t+w}] + pi_t, and the analyst
observes only the proxies lam_back = IV2 - RVb and lam_fwd = IV2 - RVf of
premium.py.  If the Section 10 statistics look alike under both hypotheses,
they cannot decide whether the ex-ante premium is reflected.

Symbols (all per session; dt = 1/252 years)
  V_t        latent annualised spot variance; log V is an AR(1) with
             compound-Poisson upward jumps (volatility spikes)
  vbar       baseline of V;  kappa_v  its mean-reversion speed (1/years)
  xi         volatility of log V;  jump_rate (per year), jump_mu, jump_sd: jump law
  eps_t      Student-t shocks (t_df degrees of freedom, unit variance)
  r_t        daily log return sqrt(V_{t-1} dt) eps_t
  pi_t       ex-ante premium, mean pi_bar, stationary sd pi_sd, speed kappa_pi
  R          number of independent synthetic samples;  n_days sessions each
This is an illustration of an identification problem, not a calibration.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import lfilter

from .diagnostics import PAPER_THRESHOLDS, conditional_table, runs_nonpositive


def simulate_panel(R: int = 300, n_days: int = 1964, window: int = 21, premium: str = "unbounded",
                   seed: int = 0, vbar: float = 0.022, kappa_v: float = 6.0, xi: float = 1.4,
                   jump_rate: float = 0.35, jump_mu: float = 1.6, jump_sd: float = 0.4,
                   pi_bar: float = 0.004, pi_sd: float = 0.004, kappa_pi: float = 15.0,
                   t_df: float = 5.0, burn: int = 300) -> dict:
    dt = 1.0 / 252.0
    N = burn + n_days + 2 * window + 1
    rng = np.random.default_rng(seed)

    # latent log-variance: x_t = phi x_{t-1} + e_t,  log V = log(vbar) + x
    phi = np.exp(-kappa_v * dt)
    counts = rng.poisson(jump_rate * dt, (R, N))
    e = xi * np.sqrt(dt) * rng.standard_normal((R, N))
    e += counts * jump_mu + np.sqrt(counts) * jump_sd * rng.standard_normal((R, N))
    x = lfilter([1.0], [1.0, -phi], e, axis=1)
    V = vbar * np.exp(x)

    # returns
    eps = rng.standard_t(t_df, (R, N)) * np.sqrt((t_df - 2.0) / t_df)
    r = np.zeros((R, N))
    r[:, 1:] = np.sqrt(V[:, :-1] * dt) * eps[:, 1:]

    # E_t[V_{t+h}] (lognormal approximation with the jump mean and variance)
    mu_step = jump_rate * dt * jump_mu
    var_step = xi ** 2 * dt + jump_rate * dt * (jump_mu ** 2 + jump_sd ** 2)
    ERV = np.zeros((R, N))
    for h in range(window):
        mean_h = phi ** h * x + mu_step * (1 - phi ** h) / (1 - phi)
        var_h = var_step * (1 - phi ** (2 * h)) / (1 - phi ** 2)
        ERV += vbar * np.exp(mean_h + 0.5 * var_h)
    ERV /= window

    # ex-ante premium
    if premium == "unbounded":
        ph = np.exp(-kappa_pi * dt)
        z = pi_sd * np.sqrt(1 - ph ** 2) * rng.standard_normal((R, N))
        pi = pi_bar + lfilter([1.0], [1.0, -ph], z, axis=1)
    elif premium == "reflected":
        s_pi = pi_sd * np.sqrt(2.0 * kappa_pi)
        pi = np.empty((R, N))
        pi[:, 0] = pi_bar
        for t in range(1, N):
            pi[:, t] = np.abs(pi[:, t - 1] + kappa_pi * (pi_bar - pi[:, t - 1]) * dt
                              + s_pi * np.sqrt(dt) * rng.standard_normal(R))
    else:
        raise ValueError("premium must be 'unbounded' or 'reflected'")

    IV2 = ERV + pi
    csum = np.concatenate([np.zeros((R, 1)), np.cumsum(r ** 2, axis=1)], axis=1)  # csum[:, k] = sum r_0..r_{k-1}
    t_idx = np.arange(burn + window, burn + window + n_days)
    RVb = (252.0 / window) * (csum[:, t_idx + 1] - csum[:, t_idx + 1 - window])
    RVf = (252.0 / window) * (csum[:, t_idx + 1 + window] - csum[:, t_idx + 1])
    return dict(pi=pi[:, t_idx], lam_back=IV2[:, t_idx] - RVb, lam_fwd=IV2[:, t_idx] - RVf,
                IV2=IV2[:, t_idx], RVb=RVb, RVf=RVf)


def summarise(panel: dict, series: str, thresholds=PAPER_THRESHOLDS) -> dict:
    X = panel[series]
    fr, mn, ep, ps, ns = [], [], [], {x: [] for x in thresholds}, {x: [] for x in thresholds}
    for row in X:
        fr.append(np.mean(row <= 0))
        mn.append(row.min())
        ep.append(len(runs_nonpositive(row)))
        tab = conditional_table(row, thresholds)
        for x, n, p in zip(tab["x"], tab["n"], tab["p"]):
            ps[x].append(p)
            ns[x].append(n)
    def band(v):
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).any():
            return (float("nan"),) * 3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (float(np.nanpercentile(v, 5)), float(np.nanmedian(v)), float(np.nanpercentile(v, 95)))
    return dict(frac_nonpos=band(fr), min=band(mn), episodes=band(ep),
                p={x: band(ps[x]) for x in thresholds}, n={x: band(ns[x]) for x in thresholds},
                share_p_defined={x: float(np.mean(np.isfinite(ps[x]))) for x in thresholds})
