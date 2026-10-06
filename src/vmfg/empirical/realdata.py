"""Real-data validation of Section 10 and of the model's testable predictions.

Constructions of the premium proxy (all annualised with m = 252, window w = 21):
  trailing  lam_back_t = IV2_t - (m/w) * sum_{j=0}^{w-1} r_{t-j}^2          (paper, Section 10)
  forward   lam_fwd_t  = IV2_t - (m/w) * sum_{j=1}^{w}  r_{t+j}^2
  intraday  lam_intra_t = IV2_t - (m/w) * sum_{j=0}^{w-1} (rv5_{t-j} + r_on_{t-j}^2)
  ex-ante   pi_hat_t   = IV2_t - F_t,  F_t an out-of-sample HAR forecast of the
            forward realised variance using information up to t only.
IV2_t = (VIX_t/100)^2.

Model predictions tested (Theorem 5.3, Theorem 5.4, corrected Section 8):
  P1 stationary OU: AR(1) coefficient below 1, drift linear in lam
  P2 Gaussian stationary law: skewness 0 and excess kurtosis 0
  P3 unbounded support: the EX-ANTE premium takes negative values
  P4 corrected Section 8: concave clearing gives Skew(lam) < 0
  P5 positive long-run premium: mean > 0
  P6 the fitted Gaussian OU reproduces the Section 10 statistics
     (share <= 0, minimum, Table 4)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .diagnostics import PAPER_THRESHOLDS, conditional_table, drift_regression, runs_nonpositive

ANN, WIN = 252.0, 21


# ---------------------------------------------------------------------------
# premium constructions
# ---------------------------------------------------------------------------
def add_constructions(df: pd.DataFrame, use_official: bool = False, window: int = WIN,
                      ann: float = ANN) -> pd.DataFrame:
    d = df.copy()
    r = d["r_official"] if use_official else d["r"]
    iv2 = (d["vix"] / 100.0) ** 2
    s = (r ** 2).rolling(window, min_periods=window).sum()
    d["IV2"] = iv2
    d["RVb"] = ann / window * s
    d["RVf"] = ann / window * s.shift(-window)
    d["lam_back"] = iv2 - d["RVb"]
    d["lam_fwd"] = iv2 - d["RVf"]
    if "rv5" in d:
        dv = d["rv5"] + d["r_on"] ** 2
        d["daily_var"] = dv
        d["RVb_intra"] = ann / window * dv.rolling(window, min_periods=window).sum()
        d["lam_intra"] = iv2 - d["RVb_intra"]
    else:
        d["daily_var"] = r ** 2
    return d


def har_exante(d: pd.DataFrame, variant: str = "levels", min_train: int = 500, refit: int = 21,
               window: int = WIN, ann: float = ANN, train_window: int | None = None,
               return_forecast: bool = False) -> pd.Series:
    """Out-of-sample HAR forecast F_t of RVf_t (the forward window t+1..t+w),
    using only pairs (x_s, RVf_s) with s <= t - w (RVf_s is known at s + w).

    variant: 'levels' (Corsi 2009); 'logs' (log-HAR, conditional-mean forecast
    with lognormal bias correction); 'logs_median' (log-HAR, conditional-median
    forecast exp(fitted)); 'levels_iv' (levels plus IV2_t as a regressor, in the
    spirit of Bekaert-Hoerova 2014).  train_window: use only the most recent
    pairs (rolling) instead of all past pairs (expanding).
    Returns pi_hat = IV2 - F (or F itself if return_forecast)."""
    dv = d["daily_var"]
    X = pd.DataFrame({"d": ann * dv, "w": ann * dv.rolling(5).mean(), "m": ann * dv.rolling(22).mean()},
                     index=d.index)
    if variant == "levels_iv":
        X["iv2"] = d["IV2"]
    y = d["RVf"]
    logs = variant in ("logs", "logs_median")
    if logs:
        X = np.log(X.clip(lower=1e-8))
        y = np.log(y.clip(lower=1e-8))
    X.insert(0, "c", 1.0)
    Xv, yv = X.to_numpy(), y.to_numpy()
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    n = len(d)
    F = np.full(n, np.nan)
    beta, s2 = None, 0.0
    for t in range(n):
        last = t - window
        if last < 0:
            continue
        if beta is None or t % refit == 0:
            idx = np.where(ok[: last + 1])[0]
            if train_window is not None:
                idx = idx[-train_window:]
            if len(idx) >= min_train:
                beta, *_ = np.linalg.lstsq(Xv[idx], yv[idx], rcond=None)
                s2 = float(np.var(yv[idx] - Xv[idx] @ beta))
        if beta is not None and np.isfinite(Xv[t]).all():
            f = float(Xv[t] @ beta)
            F[t] = np.exp(f + 0.5 * s2) if variant == "logs" else (np.exp(f) if logs else f)
    if return_forecast:
        return pd.Series(F, index=d.index, name=f"F_{variant}")
    return pd.Series(d["IV2"].to_numpy() - F, index=d.index, name=f"pi_hat_{variant}")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def stats(lam: pd.Series, dates: pd.Series | None = None, thresholds=PAPER_THRESHOLDS,
          gaps: str = "break") -> dict:
    """Section-10 statistics of a premium series on its session calendar.

    gaps="break" (default, audited): NaN marks an unobserved session; it breaks
    non-positive runs, and a pair (t, t+1) is used only if both are observed.
    gaps="close": the pre-audit convention, which drops NaN first and so pairs
    sessions across a gap; kept only to document the difference."""
    s_full = lam.copy()
    if gaps == "close":
        s_full = s_full.dropna()
    a_full = s_full.to_numpy(dtype=float)
    valid = np.isfinite(a_full)
    a = a_full[valid]
    vidx = s_full.index[valid]
    runs = runs_nonpositive(a_full)            # NaN <= 0 is False, so NaN breaks a run
    dm = a - a.mean()
    m2 = np.mean(dm ** 2)
    skew = np.mean(dm ** 3) / m2 ** 1.5
    exk = np.mean(dm ** 4) / m2 ** 2 - 3.0
    x0, x1 = a_full[:-1], a_full[1:]
    ok = np.isfinite(x0) & np.isfinite(x1)
    x0, x1 = x0[ok], x1[ok]
    phi = float(np.sum((x0 - x0.mean()) * (x1 - x1.mean())) / np.sum((x0 - x0.mean()) ** 2))
    i_min = int(np.argmin(a))
    lab = lambda i: str(pd.Timestamp(vidx[i]).date())
    return dict(n=len(a), first=lab(0), last=lab(len(a) - 1),
                n_nonpos=int(np.sum(a <= 0)), frac_nonpos=float(np.mean(a <= 0)),
                episodes=len(runs), longest=max(runs) if runs else 0,
                min=float(a[i_min]), min_date=lab(i_min), mean=float(a.mean()), sd=float(a.std()),
                skew=float(skew), exkurt=float(exk), jb=float(len(a) / 6 * (skew ** 2 + exk ** 2 / 4)),
                phi=phi, kappa_per_year=float(-ANN * np.log(phi)) if 0 < phi < 1 else float("nan"),
                gauss_frac_nonpos=float(norm.cdf(-a.mean() / a.std())),
                table=conditional_table(a_full, thresholds), gaps=gaps)


def mean_hac(lam: pd.Series, lag: int = 42) -> tuple[float, float]:
    """Mean and Newey-West standard error."""
    a = lam.dropna().to_numpy()
    e = a - a.mean()
    n = len(a)
    s = e @ e / n
    for j in range(1, lag + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return float(a.mean()), float(np.sqrt(s / n))


def fitted_ou_benchmark(st: dict, R: int = 2000, seed: int = 0, thresholds=PAPER_THRESHOLDS) -> dict:
    """Simulate the Gaussian OU of Theorem 5.3 sampled daily, with the sample's
    mean, s.d. and AR(1) coefficient, and the same n; return percentile bands
    and the rank of the observed statistics."""
    rng = np.random.default_rng(seed)
    n, m, sd, phi = st["n"], st["mean"], st["sd"], min(max(st["phi"], 0.0), 0.9999)
    x = np.empty((R, n))
    x[:, 0] = rng.standard_normal(R)
    e = rng.standard_normal((R, n)) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    x = m + sd * x
    frac = (x <= 0).mean(axis=1)
    mins = x.min(axis=1)
    now, nxt = x[:, :-1], x[:, 1:]
    ptab = {}
    for h in thresholds:
        c = now < -h
        nn = c.sum(axis=1)
        kk = (c & (nxt < -2 * h)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = np.where(nn > 0, kk / nn, np.nan)
        ptab[h] = dict(n_band=tuple(round(float(v), 1) for v in np.percentile(nn, [5, 50, 95])),
                       p_band=(tuple(round(float(v), 3) for v in np.nanpercentile(p, [5, 50, 95]))
                               if np.isfinite(p).any() else ("-",) * 3),
                       share_defined=float(np.mean(nn > 0)))
    return dict(frac_band=tuple(round(float(v), 4) for v in np.percentile(frac, [5, 50, 95])),
                min_band=tuple(round(float(v), 4) for v in np.percentile(mins, [5, 50, 95])),
                min_rank=float(np.mean(mins <= st["min"])), table=ptab)


def verdicts(st: dict, reg: pd.DataFrame, mean_se: tuple[float, float], bench: dict) -> list[tuple[str, str, str, str]]:
    """(prediction, model says, data say, verdict) rows."""
    m, se = mean_se
    rows = [
        ("P1 stationary OU (AR(1) < 1)", "phi < 1",
         f"phi = {st['phi']:.4f} (kappa = {st['kappa_per_year']:.1f}/yr, half-life {np.log(2) / (1 - st['phi']):.0f} sessions)",
         "consistent" if st["phi"] < 1 else "rejected"),
        ("P1 linear drift", "no loading on lam^2, min(lam,0)",
         f"t(lam^2) = {reg.loc['lam^2', 't']:.2f}, t(min) = {reg.loc['min(lam,0)', 't']:.2f}",
         "consistent" if max(abs(reg.loc['lam^2', 't']), abs(reg.loc['min(lam,0)', 't'])) < 1.96 else "rejected"),
        ("P2 Gaussian law", "skew 0, excess kurtosis 0",
         f"skew {st['skew']:+.2f}, excess kurtosis {st['exkurt']:.1f}, JB = {st['jb']:.0f}",
         "rejected" if st["jb"] > 5.99 else "consistent"),
        ("P2 share <= 0 implied by mean/sd", f"Phi(-mean/sd) = {st['gauss_frac_nonpos']:.3f}",
         f"{st['frac_nonpos']:.3f}", "consistent" if abs(st["gauss_frac_nonpos"] - st["frac_nonpos"]) < 0.05 else "rejected"),
        ("P6 minimum under fitted OU", f"5-95%: {bench['min_band'][0]:.3f} to {bench['min_band'][2]:.3f}",
         f"{st['min']:.3f} (rank {bench['min_rank']:.4f})",
         "consistent" if bench["min_band"][0] <= st["min"] <= bench["min_band"][2] else "rejected"),
        ("P4 corrected Section 8 sign", "Skew(lam) < 0 (draft said > 0)", f"skew {st['skew']:+.2f}",
         "consistent with corrected sign" if st["skew"] < 0 else "consistent with draft sign"),
        ("P5 positive long-run premium", "mean > 0", f"mean {m:.4f} (NW s.e. {se:.4f}, t = {m / se:.1f})",
         "consistent" if m / se > 1.96 else ("rejected" if m / se < -1.96 else "inconclusive")),
    ]
    return rows


# ---------------------------------------------------------------------------
# regressions with Newey-West standard errors, and the mechanism check
# ---------------------------------------------------------------------------
def ols_nw(y: pd.Series, X: pd.DataFrame, lag: int) -> pd.DataFrame:
    Z = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = Z["_y"].to_numpy()
    Xv = np.column_stack([np.ones(len(Z)), Z[X.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ beta
    XtXi = np.linalg.inv(Xv.T @ Xv)
    Xe = Xv * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lag + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return pd.DataFrame(dict(coef=beta, nw_se=se, t=beta / se, n=len(Z)), index=["const", *X.columns])


def ar1_phi(x: pd.Series) -> float:
    a = x.dropna().to_numpy()
    x0, x1 = a[:-1], a[1:]
    return float(np.sum((x0 - x0.mean()) * (x1 - x1.mean())) / np.sum((x0 - x0.mean()) ** 2))


def mechanism_table(weekly: pd.DataFrame, premia: list[str], proxies: list[str], lag: int = 8) -> pd.DataFrame:
    """Slope of each premium measure on each deployment proxy, in levels and in
    weekly changes.  Model (Ass. 2.1, constant intercept a): lam = a - b Y with
    b > 0, i.e. a negative slope and correlation -1."""
    rows = []
    for p in premia:
        for x in proxies:
            lv = ols_nw(weekly[p], weekly[[x]], lag)
            ch = ols_nw(weekly[p].diff(), weekly[[x]].diff(), lag)
            rows.append(dict(premium=p, proxy=x,
                             corr_levels=float(weekly[[p, x]].dropna().corr().iloc[0, 1]),
                             t_levels=float(lv.loc[x, "t"]),
                             corr_changes=float(weekly[[p, x]].diff().dropna().corr().iloc[0, 1]),
                             t_changes=float(ch.loc[x, "t"]), n=int(lv.loc[x, "n"])))
    return pd.DataFrame(rows)
