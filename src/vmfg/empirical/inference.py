"""Statistical inference for the empirical section.

Every reported statistic gets a standard error by a method suited to its
dependence structure:

  * marginal statistics of a persistent daily series (share <= 0, mean, s.d.,
    skewness, excess kurtosis, the gap to the Gaussian share): moving-block
    bootstrap of the observations, block length L sessions (Kunsch 1989);
  * pair statistics (AR(1) coefficient, Table-4 counts and conditional
    probabilities): moving-block bootstrap of consecutive valid pairs
    (x_t, x_{t+1}), so that a resampled pair is always a true pair;
  * the mean: Newey-West HAC standard error (Newey-West 1987);
  * mean reversion: augmented Dickey-Fuller t-statistic with p lags, compared
    with the 5% asymptotic critical value -2.86 (MacKinnon 1994, constant);
  * the annualised speed kappa = -m log(phi): delta method,
    s.e.(kappa) = m s.e.(phi)/phi;
  * simulation-based benchmarks: Monte Carlo p-value (1 + #{T* <= T})/(R + 1)
    and the binomial Monte Carlo standard error of simulated tail frequencies.

Episode counts and the longest episode are path functionals that block
resampling cuts at block boundaries; they are assessed against the fitted
Ornstein-Uhlenbeck benchmark instead, with Monte Carlo p-values.

Symbols: x a series possibly containing NaN (unobserved sessions); L block
length; B bootstrap replications; m annualisation factor (252 daily, 52 weekly).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

ADF_CRIT_5PCT = -2.86


def block_indices(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices of length n (blocks of length L)."""
    L = max(1, min(L, n))
    nb = -(-n // L)
    starts = rng.integers(0, n - L + 1, nb)
    return (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]


def valid_pairs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive pairs (x_t, x_{t+1}) with both observed, in time order."""
    x = np.asarray(x, dtype=float)
    x0, x1 = x[:-1], x[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    return x0[ok], x1[ok]


def marginal_stats(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    m, s = a.mean(), a.std()
    d = a - m
    m2 = np.mean(d ** 2)
    share = float(np.mean(a <= 0))
    gshare = float(norm.cdf(-m / s))
    return dict(share=share, mean=float(m), sd=float(s),
                skew=float(np.mean(d ** 3) / m2 ** 1.5), exkurt=float(np.mean(d ** 4) / m2 ** 2 - 3),
                gauss_share=gshare, gauss_gap=share - gshare)


def phi_hat(x0: np.ndarray, x1: np.ndarray) -> float:
    return float(np.sum((x0 - x0.mean()) * (x1 - x1.mean())) / np.sum((x0 - x0.mean()) ** 2))


def pair_stats(x0: np.ndarray, x1: np.ndarray, thresholds) -> dict:
    out = dict(phi=phi_hat(x0, x1))
    for h in thresholds:
        c = x0 < -h
        n = int(c.sum())
        out[f"n_{h}"] = n
        out[f"p_{h}"] = float(np.mean(x1[c] < -2 * h)) if n else np.nan
    return out


def bootstrap_series(x: np.ndarray, thresholds, L: int = 63, B: int = 2000, seed: int = 0) -> dict:
    """Point estimates and block-bootstrap standard errors / 95% percentile
    intervals for marginal and pair statistics of x (NaN = unobserved)."""
    x = np.asarray(x, dtype=float)
    a = x[np.isfinite(x)]
    x0, x1 = valid_pairs(x)
    est = {**marginal_stats(a), **pair_stats(x0, x1, thresholds)}
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in est}
    na, npair = len(a), len(x0)
    for _ in range(B):
        ia = block_indices(na, L, rng)
        for k, v in marginal_stats(a[ia]).items():
            reps[k].append(v)
        ip = block_indices(npair, L, rng)
        for k, v in pair_stats(x0[ip], x1[ip], thresholds).items():
            reps[k].append(v)
    se, lo, hi = {}, {}, {}
    for k, v in reps.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        se[k] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        lo[k], hi[k] = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
    return dict(est=est, se=se, lo=lo, hi=hi, L=L, B=B)


def nw_mean(x: np.ndarray, lag: int) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    e = a - a.mean()
    n = len(a)
    s = e @ e / n
    for j in range(1, lag + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return float(a.mean()), float(np.sqrt(s / n))


def adf(x: np.ndarray, p: int = 21) -> dict:
    """ADF regression  dx_t = alpha + gamma x_{t-1} + sum_{j=1..p} delta_j dx_{t-j} + e_t
    on the longest run of consecutive observed values' concatenation is not used:
    NaNs are removed and the regression is run on the observed series in order."""
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    dx = np.diff(a)
    y = dx[p:]
    cols = [np.ones_like(y), a[p:-1]] + [dx[p - j:-j] for j in range(1, p + 1)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    s2 = e @ e / (len(y) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    return dict(gamma=float(beta[1]), t=float(t), crit5=ADF_CRIT_5PCT, lags=p,
                reject_unit_root=bool(t < ADF_CRIT_5PCT))


def kappa_delta(phi: float, se_phi: float, m: float) -> tuple[float, float]:
    """kappa = -m log(phi) and its delta-method standard error."""
    return float(-m * np.log(phi)), float(m * se_phi / phi)


def mc_pvalue_lower(sim: np.ndarray, obs: float) -> float:
    """One-sided Monte Carlo p-value for 'obs is small': (1 + #{sim <= obs})/(R + 1)."""
    sim = np.asarray(sim, dtype=float)
    return float((1 + np.sum(sim <= obs)) / (len(sim) + 1))


def mc_pvalue_two_sided(sim: np.ndarray, obs: float) -> float:
    lo = mc_pvalue_lower(sim, obs)
    hi = float((1 + np.sum(np.asarray(sim) >= obs)) / (len(sim) + 1))
    return float(min(1.0, 2 * min(lo, hi)))


def bootstrap_mean_indicator(ind: np.ndarray, L: int, B: int, seed: int = 0) -> tuple[float, float]:
    """Share of a 0/1 series and its block-bootstrap standard error."""
    ind = np.asarray(ind, dtype=float)
    ind = ind[np.isfinite(ind)]
    rng = np.random.default_rng(seed)
    reps = [ind[block_indices(len(ind), L, rng)].mean() for _ in range(B)]
    return float(ind.mean()), float(np.std(reps, ddof=1))


def bootstrap_median(x: np.ndarray, L: int, B: int, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    reps = [np.median(x[block_indices(len(x), L, rng)]) for _ in range(B)]
    return float(np.median(x)), float(np.std(reps, ddof=1))
