#!/usr/bin/env python3
"""12 - Is the premium LOCALLY a Gaussian Ornstein-Uhlenbeck process?

The clearing relation is a local linearisation, so the model's testable content is
local: over a short step D the transition law is
    lam_{t+D} | lam_t  ~  N( lbar + (lam_t - lbar) e^{-kappa D},  b^2 sigma0^2 (1 - e^{-2 kappa D}) / (2 kappa) ),
i.e. Gaussian, homoskedastic, with drift linear in the level and independent
innovations.  We test exactly that, on the discretely sampled process:

  T1 transition law, full sample: AR(1) by OLS on observed consecutive pairs;
     standardised residuals z; normality (Shapiro-Wilk, Jarque-Bera), independence
     (Ljung-Box on z and z^2), homoskedasticity (Breusch-Pagan on z^2 vs lam_t),
     probability-integral transform u = Phi(z) (Diebold-Gunther-Tay 1998).
  T2 local windows: non-overlapping windows of W sessions (W = 21, 63); in each,
     AR(1) fit, Shapiro-Wilk on residuals, Ljung-Box on residuals, RESET t-test
     for a lam^2 term. Rejection rates at 5% against the nominal 5%.
  T3 size and power of T2 by simulation: Gaussian AR(1) windows (size) and windows
     whose innovations are resampled from the pooled standardised residuals (power
     against the observed tails).
  T4 horizons D = 1, 5, 10, 21: excess kurtosis of non-overlapping D-step residuals.

Series: the trailing proxy lam_back; the ex-ante estimate pi_hat = IV2 - F (the
conditional-median log-HAR forecast, the best calibrated in Table exante); and IV2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import inference as F, realdata as RD  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
END = pd.Timestamp("2026-08-26")
rng = np.random.default_rng(2026)


def ar1(x0, x1):
    X = np.column_stack([np.ones_like(x0), x0])
    beta, *_ = np.linalg.lstsq(X, x1, rcond=None)
    e = x1 - X @ beta
    s = np.sqrt(e @ e / max(len(e) - 2, 1))
    return beta, e, s


def ljung_box(z, m):
    z = z - z.mean()
    n = len(z)
    denom = z @ z
    q = 0.0
    for k in range(1, m + 1):
        rk = (z[k:] @ z[:-k]) / denom
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    return float(q), float(stats.chi2.sf(q, m))


def breusch_pagan(z, x0):
    y = z ** 2
    X = np.column_stack([np.ones_like(x0), x0])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    lm = len(y) * r2
    return float(lm), float(stats.chi2.sf(lm, 1))


def reset_t(x0, x1):
    X = np.column_stack([np.ones_like(x0), x0, x0 ** 2])
    beta, *_ = np.linalg.lstsq(X, x1, rcond=None)
    e = x1 - X @ beta
    s2 = e @ e / max(len(e) - 3, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    t = beta[2] / np.sqrt(max(cov[2, 2], 1e-300))
    return float(2 * stats.t.sf(abs(t), max(len(e) - 3, 1)))


def transition_tests(x):
    x0, x1 = F.valid_pairs(x)
    beta, e, s = ar1(x0, x1)
    z = e / s
    u = stats.norm.cdf(z)
    sw = stats.shapiro(z[:5000])
    jb = stats.jarque_bera(z)
    return dict(n=len(z), phi=beta[1], exkurt=float(stats.kurtosis(z)), skew=float(stats.skew(z)),
                sw_p=float(sw.pvalue), jb=float(jb.statistic), jb_p=float(jb.pvalue),
                lb_z_p=ljung_box(z, 10)[1], lb_z2_p=ljung_box(z ** 2, 10)[1], bp_p=breusch_pagan(z, x0)[1],
                ks_pit_p=float(stats.kstest(u, "uniform").pvalue), z=z, phi_s=(beta, s))


def windows(series: pd.Series, W: int):
    """Non-overlapping calendar windows of W+1 sessions; per window the valid pairs."""
    a = series.to_numpy(dtype=float)
    idx = series.index
    out = []
    for start in range(0, len(a) - W, W):
        seg = a[start:start + W + 1]
        x0, x1 = F.valid_pairs(seg)
        if len(x0) >= W - 2:
            out.append((idx[start], x0, x1))
    return out


def window_tests(series: pd.Series, W: int):
    rows = []
    lb_m = 3 if W <= 21 else 5
    for d0, x0, x1 in windows(series, W):
        beta, e, s = ar1(x0, x1)
        if s == 0:
            continue
        rows.append(dict(start=d0, sw_p=float(stats.shapiro(e / s).pvalue), lb_p=ljung_box(e / s, lb_m)[1],
                         reset_p=reset_t(x0, x1), phi=beta[1]))
    return pd.DataFrame(rows)


def simulate_rejection(phi, W, n_sim, innov, lb_m):
    """Rejection rates of the window tests for AR(1) windows with given innovations sampler."""
    rej = dict(sw=0, lb=0, reset=0)
    for _ in range(n_sim):
        eps = innov(W + 1)
        x = np.empty(W + 1)
        x[0] = eps[0] / np.sqrt(max(1 - phi ** 2, 1e-6))
        for t in range(1, W + 1):
            x[t] = phi * x[t - 1] + eps[t]
        beta, e, s = ar1(x[:-1], x[1:])
        rej["sw"] += stats.shapiro(e / s).pvalue < 0.05
        rej["lb"] += ljung_box(e / s, lb_m)[1] < 0.05
        rej["reset"] += reset_t(x[:-1], x[1:]) < 0.05
    return {k: v / n_sim for k, v in rej.items()}


def horizon_kurtosis(series: pd.Series, D: int):
    a = series.to_numpy(dtype=float)
    x0, x1 = a[:-D:D], a[D::D]
    ok = np.isfinite(x0) & np.isfinite(x1)
    beta, e, s = ar1(x0[ok], x1[ok])
    z = e / s
    return float(stats.kurtosis(z)), int(len(z)), float(stats.jarque_bera(z).pvalue)


def main():
    india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
    d = RD.add_constructions(india)
    us = pd.read_csv(PROC / "us_daily_github.csv", parse_dates=["date"]).set_index("date")
    du = RD.add_constructions(us)
    lb = d["lam_back"].loc[:END]
    v = lb.dropna()
    span = (v.index[-1964], v.index[-1])
    Fi = RD.har_exante(d, "logs_median", return_forecast=True)
    Fu = RD.har_exante(du, "logs_median", return_forecast=True)
    series = {
        ("NIFTY", "proxy lam_back"): d["lam_back"].loc[span[0]:span[1]],
        ("NIFTY", "ex-ante pi_hat"): (d["IV2"] - Fi).loc[span[0]:span[1]],
        ("NIFTY", "IV2"): d["IV2"].loc[span[0]:span[1]],
        ("US", "proxy lam_back"): du["lam_back"],
        ("US", "ex-ante pi_hat"): (du["IV2"] - Fu).loc["2008-01-01":],
        ("US", "IV2"): du["IV2"],
    }
    t1_rows, t2_rows, t3_rows, t4_rows, rejecting = [], [], [], [], []
    for (mk, name), s in series.items():
        s = s.loc[s.first_valid_index():]
        tt = transition_tests(s.to_numpy(dtype=float))
        t1_rows.append(dict(market=mk, series=name, **{k: v for k, v in tt.items() if k not in ("z", "phi_s")}))
        (beta, sig), z = tt["phi_s"], tt["z"]
        for W in (21, 63):
            wt = window_tests(s, W)
            nwin = len(wt)
            se_nom = np.sqrt(0.05 * 0.95 / nwin)
            rates = {k: float((wt[f"{k}_p"] < 0.05).mean()) for k in ("sw", "lb", "reset")}
            allpass = float(((wt.sw_p >= 0.05) & (wt.lb_p >= 0.05) & (wt.reset_p >= 0.05)).mean())
            t2_rows.append(dict(market=mk, series=name, W=W, windows=nwin, rej_normality=rates["sw"],
                                rej_independence=rates["lb"], rej_linearity=rates["reset"],
                                pass_all=allpass, se_under_null=se_nom))
            if W == 21:
                bad = wt[wt.sw_p < 0.05]
                rejecting.append(dict(market=mk, series=name, starts=", ".join(dd.strftime("%Y-%m") for dd in bad.start[:40])))
            lb_m = 3 if W <= 21 else 5
            phi = float(np.clip(beta[1], -0.99, 0.99))
            size = simulate_rejection(phi, W, 1000, lambda n: rng.standard_normal(n), lb_m)
            power = simulate_rejection(phi, W, 1000, lambda n: rng.choice(z, n, replace=True), lb_m)
            t3_rows.append(dict(market=mk, series=name, W=W, size_sw=size["sw"], size_lb=size["lb"], size_reset=size["reset"],
                                power_sw=power["sw"], power_lb=power["lb"], power_reset=power["reset"]))
        for D in (1, 5, 10, 21):
            k, n, p = horizon_kurtosis(s, D)
            t4_rows.append(dict(market=mk, series=name, D=D, n=n, exkurt=k, jb_p=p))
    T1, T2, T3, T4 = (pd.DataFrame(r) for r in (t1_rows, t2_rows, t3_rows, t4_rows))
    REJ = pd.DataFrame(rejecting)
    for nm, df in (("T1", T1), ("T2", T2), ("T3", T3), ("T4", T4), ("rejecting_windows", REJ)):
        df.to_csv(RES / f"12_local_ou_{nm}.csv", index=False)
    pd.set_option("display.width", 220)
    txt = ["# 12 - Local Gaussian-OU tests", "", "## T1 transition law (full sample)", T1.round(4).to_string(), "",
           "## T2 local windows (rejection rates at 5%)", T2.round(3).to_string(), "",
           "## T3 size and power of the window tests", T3.round(3).to_string(), "",
           "## T4 excess kurtosis of D-step residuals", T4.round(3).to_string(), "",
           "## windows rejecting normality (W = 21), start months", REJ.to_string()]
    (RES / "12_local_ou.md").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
