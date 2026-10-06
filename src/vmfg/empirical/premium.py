"""Variance-premium proxies built from daily closes and India VIX.

Symbols
  r_t          daily log return log(close_t / close_{t-1})
  w            window length in sessions (paper: 21)
  m            annualisation factor (sessions per year; default 252)
  IV2_t        (VIX_t / 100)^2, annualised model-free implied variance
               (India VIX is a 30-calendar-day NIFTY option measure)
  RVb_t        backward realised variance  (m/w) * sum_{j=0}^{w-1} r_{t-j}^2
               -- known at the close of day t (the dashboard's "Lambda_ante")
  RVf_t        forward realised variance   (m/w) * sum_{j=1}^{w} r_{t+j}^2
               -- the ex-post counterpart of IV2_t (Carr-Wu convention)
  lam_back_t   IV2_t - RVb_t
  lam_fwd_t    IV2_t - RVf_t

Neither proxy is the model's lam_t, which is an EX-ANTE premium
E^Q[RV] - E^P[RV].  lam_fwd = (ex-ante premium) - (forecast error of RV),
and lam_back adds a further lag error.  Negative values of either proxy are
therefore compatible with an ex-ante premium that never goes negative
(see identification.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def realized_variance(r: pd.Series, window: int = 21, ann: float = 252.0,
                      direction: str = "backward") -> pd.Series:
    s = (r ** 2).rolling(window, min_periods=window).sum()
    if direction == "backward":
        out = s
    elif direction == "forward":
        out = s.shift(-window)
    else:
        raise ValueError("direction must be 'backward' or 'forward'")
    return (ann / window) * out


def implied_variance(vix_percent: pd.Series) -> pd.Series:
    return (vix_percent.astype(float) / 100.0) ** 2


def build_premia(df: pd.DataFrame, window: int = 21, ann: float = 252.0) -> pd.DataFrame:
    """Add r, IV2, RVb, RVf, lam_back, lam_fwd to a validated frame."""
    out = df.copy()
    out["r"] = log_returns(out["close"])
    out["IV2"] = implied_variance(out["vix"])
    out["RVb"] = realized_variance(out["r"], window, ann, "backward")
    out["RVf"] = realized_variance(out["r"], window, ann, "forward")
    out["lam_back"] = out["IV2"] - out["RVb"]
    out["lam_fwd"] = out["IV2"] - out["RVf"]
    return out
