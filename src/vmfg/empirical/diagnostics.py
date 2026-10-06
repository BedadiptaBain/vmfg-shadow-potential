"""Section 10 statistics, their uncertainty, and checks that need no data.

Symbols
  lam           a premium series (lam_back or lam_fwd), one value per session
  x             excursion threshold (the paper uses 0.02, 0.05, 0.10, 0.20)
  n_x           number of sessions t with lam_t < -x whose successor t+1 exists
  k_x           number of those sessions with lam_{t+1} < -2x
  p_x = k_x/n_x the conditional probability reported in Table 4
  L             block length of the moving-block bootstrap (sessions)
  B             number of bootstrap replications
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

PAPER_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)
PAPER_TABLE4_CURRENT = {0.02: (90, 0.556), 0.05: (46, 0.457), 0.10: (22, 0.682), 0.20: (15, 0.467)}
PAPER_TABLE4_PREVIOUS = {0.02: (389, 0.556), 0.05: (232, 0.457), 0.10: (129, 0.682), 0.20: (45, 0.467)}
PAPER_N = 1964
PAPER_FRAC_NONPOS = 0.175
PAPER_MIN = -0.535


# --------------------------------------------------------------------------
# the statistics of Section 10
# --------------------------------------------------------------------------
def runs_nonpositive(lam: np.ndarray) -> list[int]:
    """Lengths of maximal runs of consecutive sessions with lam <= 0."""
    runs, cur = [], 0
    for m in (np.asarray(lam) <= 0):
        if m:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def conditional_table(lam: np.ndarray, thresholds=PAPER_THRESHOLDS) -> pd.DataFrame:
    a = np.asarray(lam, dtype=float)
    now, nxt = a[:-1], a[1:]
    ok = np.isfinite(now) & np.isfinite(nxt)
    rows = []
    for x in thresholds:
        cond = ok & (now < -x)
        n = int(cond.sum())
        k = int((cond & (nxt < -2 * x)).sum())
        rows.append(dict(x=x, n=n, k=k, p=(k / n if n else np.nan)))
    return pd.DataFrame(rows)


def section10_stats(lam: pd.Series, thresholds=PAPER_THRESHOLDS) -> dict:
    s = lam.dropna()
    a = s.to_numpy()
    runs = runs_nonpositive(a)
    d = a - a.mean()
    return dict(
        n=len(a), first=str(s.index[0]) if len(s) else "", last=str(s.index[-1]) if len(s) else "",
        frac_nonpos=float(np.mean(a <= 0)), n_nonpos=int(np.sum(a <= 0)),
        episodes=len(runs), longest=max(runs) if runs else 0,
        min=float(a.min()), mean=float(a.mean()), sd=float(a.std()),
        skew=float(np.mean(d ** 3) / np.mean(d ** 2) ** 1.5),
        exkurt=float(np.mean(d ** 4) / np.mean(d ** 2) ** 2 - 3.0),
        table=conditional_table(a, thresholds),
    )


def block_bootstrap(lam: np.ndarray, L: int = 21, B: int = 2000, seed: int = 0,
                    thresholds=PAPER_THRESHOLDS) -> dict:
    """Moving-block bootstrap 90% intervals for the fraction of non-positive
    sessions and for each p_x (pairs (lam_t, lam_{t+1}) are the resampled units)."""
    a = np.asarray(lam, dtype=float)
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    npair = len(a) - 1
    nblocks = int(math.ceil(npair / L))
    fr, ps = [], {x: [] for x in thresholds}
    for _ in range(B):
        starts = rng.integers(0, npair - L + 1, nblocks)
        idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:npair]
        now, nxt = a[idx], a[idx + 1]
        fr.append(np.mean(now <= 0))
        for x in thresholds:
            c = now < -x
            ps[x].append(np.mean(nxt[c] < -2 * x) if c.any() else np.nan)
    def q(v):
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).any():
            return (float("nan"), float("nan"))
        lo, hi = np.nanpercentile(v, [5, 95])
        return (round(float(lo), 4), round(float(hi), 4))
    return dict(frac_nonpos_90=q(fr), p_90={x: q(ps[x]) for x in thresholds},
                p_undefined_share={x: float(np.mean(np.isnan(ps[x]))) for x in thresholds}, L=L, B=B)


def drift_regression(lam: pd.Series, lag: int = 25) -> pd.DataFrame:
    """OLS of d lam_{t+1} = lam_{t+1} - lam_t on [1, lam_t, lam_t^2, min(lam_t, 0)]
    with Newey-West (Bartlett) standard errors.  Under Theorem 5.3 the drift is
    linear, so only the constant and lam_t should load; a barrier-like push
    away from zero would load on the hinge min(lam_t, 0).  This is a shape
    diagnostic for the proxy only: the proxy is not the ex-ante premium."""
    a = lam.dropna().to_numpy()
    y = a[1:] - a[:-1]
    l = a[:-1]
    X = np.column_stack([np.ones_like(l), l, l ** 2, np.minimum(l, 0.0)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lag + 1):
        w = 1.0 - j / (lag + 1.0)
        G = Xe[j:].T @ Xe[:-j]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return pd.DataFrame(dict(coef=beta, nw_se=se, t=beta / se),
                        index=["const", "lam", "lam^2", "min(lam,0)"])


# --------------------------------------------------------------------------
# consistency checks on the numbers printed in the paper (no data needed)
# --------------------------------------------------------------------------
def lattice_ks(p: float, n: int, decimals: int = 3) -> list[int]:
    """Integers k with round(k/n, decimals) == p.  Empty list => (p, n) impossible."""
    return [k for k in range(n + 1) if round(k / n, decimals) == round(p, decimals)]


def table4_consistency() -> pd.DataFrame:
    n_nonpos_max = int(math.floor((PAPER_FRAC_NONPOS + 0.0005) * PAPER_N))
    rows = []
    for label, tab in (("current draft", PAPER_TABLE4_CURRENT), ("previous draft", PAPER_TABLE4_PREVIOUS)):
        for x, (n, p) in tab.items():
            ks = lattice_ks(p, n)
            rows.append(dict(draft=label, x=x, n=n, p=p, attainable_k=ks or "none",
                             n_exceeds_nonpositive_sessions=n > n_nonpos_max))
    return pd.DataFrame(rows)


def weekday_count(start: str, end: str) -> int:
    return int(np.busday_count(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D")))


def session_count_range(start: str, end: str, holidays_per_year=(11, 17)) -> tuple[int, int]:
    """Range of NSE sessions between two dates: weekdays minus 11-17 weekday
    holidays per year (NSE lists typically have 12-16 weekday closures)."""
    wd = weekday_count(start, end)
    years = (np.datetime64(end) - np.datetime64(start)).astype(int) / 365.25
    return int(wd - holidays_per_year[1] * years), int(wd - holidays_per_year[0] * years)


def gaussian_min_zscore(mean_premium: float, frac_nonpos: float = PAPER_FRAC_NONPOS,
                        sample_min: float = PAPER_MIN) -> float:
    """If lam were Gaussian (Theorem 5.3) with the given mean and with
    P(lam <= 0) = frac_nonpos, its sd is mean / Phi^{-1}(1 - frac_nonpos);
    return the z-score of the sample minimum under that law."""
    zq = norm.ppf(1.0 - frac_nonpos)
    sd = mean_premium / zq
    return (sample_min - mean_premium) / sd
