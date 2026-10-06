#!/usr/bin/env python3
"""13 - Two-factor (stochastic demand) equilibrium: verification and confrontation with data.

Model: lam_t = a_t - b Y_t with a_t = abar + alpha_t, d alpha = -kappa_a alpha dt + sigma_a dW^a
(W^a independent of W^0 and of idiosyncratic noise). Claim (Proposition in Section 6):
    u(Y, a) = A Y + C a + B,  A = eta (kappa_ind - kappa),  C = 1/(rho + kappa + kappa_a),
    B = kappa_a abar C / (rho + kappa),
    dY = [ -kappa (Y - Ybar) + (C/eta) alpha ] dt + sigma0 dW^0,   Ybar = abar/(chi + b).
Scaled deviations: ytil = b (Y - Ybar), beta = b C / eta, r = sigma_a / (b sigma0):
    d ytil = (-kappa ytil + beta alpha) dt + dW^0 (unit supply noise), d alpha = -kappa_a alpha dt + r dW^a,
    lam - lbar = alpha - ytil.
Moments (weekly, D = 1/52): premium speed, deployment speed, correlation in levels and in weekly changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import inference as F, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402
from vmfg import twofactor as TF  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
D = 1.0 / 52.0
lines = ["# 13 - Two-factor equilibrium", ""]

# ------------------------------------------------------------------ (1) symbolic verification
Y, a, x = sp.symbols("Y a x", real=True)
eta, rho, chi, b, s0, sa, ka, abar, sig = sp.symbols("eta rho chi b sigma_0 sigma_a kappa_a abar sigma", positive=True)
ki, k = sp.symbols("kappa_ind kappa", positive=True)
q = -eta * ki / 2
A = eta * (ki - k)
Cc = 1 / (rho + k + ka)
Bb = ka * abar * Cc / (rho + k)
u = A * Y + Cc * a + Bb
w = sp.Function("w")(Y, a)
V = q * x ** 2 + x * u + w
bhat = (u + 2 * q * Y) / eta
Lam = a - b * Y
H = (-rho * V + V.diff(x) ** 2 / (2 * eta) + bhat * V.diff(Y) + (-ka * (a - abar)) * V.diff(a)
     + (sig ** 2 + s0 ** 2) / 2 * V.diff(x, 2) + s0 ** 2 / 2 * V.diff(Y, 2) + sa ** 2 / 2 * V.diff(a, 2)
     + s0 ** 2 * V.diff(x).diff(Y) + Lam * x - chi / 2 * x ** 2)
sub = {chi: eta * ki * (ki + rho), b: eta * k * (k + rho) - eta * ki * (ki + rho)}
He = sp.expand(H.subs(sub))
ok2 = sp.simplify(He.coeff(x, 2)) == 0
c1 = sp.expand(He.coeff(x, 1))
ok1 = all(sp.simplify(c1.coeff(Y, i).coeff(a, j)) == 0 for i in range(2) for j in range(2))
drift = sp.simplify((bhat + k * (Y - abar / (eta * k * (k + rho))) - (Cc / eta) * (a - abar)).subs(sub))
lines += [f"- x^2 equation holds: {ok2}; x^1 equation holds identically in (Y, a): {ok1}; "
          f"aggregate drift equals -kappa (Y - Ybar) + (C/eta)(a - abar) with Ybar = abar/(chi+b): {drift == 0}", ""]


# ------------------------------------------------------------------ (2) implied moments
def model_moments(kap, kap_a, beta, r, averaged=True):
    """Model moments of weekly MEANS of daily values (averaged=True), as in the data."""
    mm = TF.moments(kap, kap_a, beta, r, averaged=averaged)
    return np.array([mm["k_lam"], mm["k_Y"], mm["corr_lev"], mm["corr_chg_1"]])


def empirical_moments(W: np.ndarray) -> np.ndarray:
    lam, y = W[:, 0], W[:, 1]
    k_l = -np.log(F.phi_hat(lam[:-1], lam[1:])) / D
    k_y = -np.log(F.phi_hat(y[:-1], y[1:])) / D
    r_lev = np.corrcoef(lam, y)[0, 1]
    r_chg = np.corrcoef(np.diff(lam), np.diff(y))[0, 1]
    return np.array([k_l, k_y, r_lev, r_chg])


RHO0 = 0.05


def beta_of(kap, kap_a, s_share):
    """beta = b C / eta with b/eta = s * kappa (kappa + rho), s = b/(chi + b) in (0, 1): chi >= 0 enforced."""
    return s_share * kap * (kap + RHO0) / (RHO0 + kap + kap_a)


def fit(target, se=None, x0=(3.0, 15.0, 0.5, 5.0), averaged=True):
    """Constrained method of moments over (kappa, kappa_a, s, r) with s = b/(chi+b) in (0, 1)."""
    scale = se if se is not None else np.array([5.0, 1.0, 0.1, 0.05])
    obj = lambda p: (model_moments(p[0], p[1], beta_of(p[0], p[1], p[2]), p[3], averaged) - target) / scale
    best = None
    for start in (x0, (6.0, 12.0, 0.9, 6.0), (2.5, 20.0, 0.99, 10.0), (4.0, 8.0, 0.7, 3.0)):
        res = least_squares(obj, start, bounds=([0.05, 0.05, 1e-4, 0.01], [60.0, 200.0, 1.0, 200.0]))
        if best is None or res.cost < best.cost:
            best = res
    p = best.x
    return p, model_moments(p[0], p[1], beta_of(p[0], p[1], p[2]), p[3], averaged), best.cost


india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
m = d[["lam_back"]].join(poi[["log_short_detr"]], how="inner")
wk = m.resample("W-FRI").mean().dropna()
import os as _os
OI_END = _os.environ.get("VMFG_OI_END")
TAG = _os.environ.get("VMFG_TAG", "")
if OI_END:
    wk = wk.loc[:OI_END]
Wd = wk[["lam_back", "log_short_detr"]].to_numpy()
target = empirical_moments(Wd)
rng = np.random.default_rng(13)
reps_t = []
for _ in range(500):
    tb = empirical_moments(Wd[F.block_indices(len(Wd), 8, rng)])
    if np.all(np.isfinite(tb)):
        reps_t.append(tb)
reps_t = np.array(reps_t)
se_t = np.std(reps_t, axis=0, ddof=1)
p_hat, m_hat, cost = fit(target, se_t)
p_pt, m_pt, cost_pt = fit(target, se_t, averaged=False)
reps_p = []
for tb in reps_t[:300]:
    pb, _, _ = fit(tb, se_t, x0=p_hat)
    reps_p.append(pb)
reps_p = np.array(reps_p)
se_p = np.std(reps_p, axis=0, ddof=1)
lo_p, hi_p = np.percentile(reps_p, [5, 95], axis=0)
zfit = (m_hat - target) / se_t
lab_m = ["premium speed (1/yr)", "deployment speed (1/yr)", "corr, levels", "corr, weekly changes"]
lab_p = ["kappa (1/yr)", "kappa_a (1/yr)", "s = b/(chi+b)", "r = sigma_a/(b sigma0)"]
mom = pd.DataFrame(dict(moment=lab_m, data=target, se=se_t, fitted=m_hat, z=zfit))
par = pd.DataFrame(dict(parameter=lab_p, estimate=p_hat, se=se_p, q05=lo_p, q95=hi_p))
# one-factor benchmark: alpha = 0  -> premium and deployment share the speed and have correlation -1
one = model_moments(p_hat[0], p_hat[1], beta_of(p_hat[0], p_hat[1], p_hat[2]), 1e-6)
lines += ["## Method-of-moments fit to NIFTY weekly data (premium proxy vs detrended log short OI)", "",
          mom.round(4).to_string(index=False), "", par.round(4).to_string(index=False), "",
          f"- fit cost {cost:.2e}; bootstrap replications {len(reps_p)} (block length 8 weeks)",
          f"- one-factor limit (sigma_a -> 0) at the fitted speeds: {np.round(one, 3).tolist()}", ""]

lines += [f"- constrained fit (chi >= 0 imposed through s = b/(chi+b) <= 1, rho = {RHO0}), model moments of weekly means; "
          f"sum of squared z-scores {np.sum(zfit ** 2):.3f}",
          f"- for comparison, fitting point-sampled model moments (the v6 approximation): params {np.round(p_pt, 3).tolist()}, "
          f"fitted {np.round(m_pt, 4).tolist()}, sum of squared z {np.sum(((m_pt - target) / se_t) ** 2):.3f}", ""]

# ------------------------------------------------------------------ (3) ex-ante premium skewness
Fi = RD.har_exante(d, "logs_median", return_forecast=True)
us = pd.read_csv(PROC / "us_daily_github.csv", parse_dates=["date"]).set_index("date")
du = RD.add_constructions(us)
Fu = RD.har_exante(du, "logs_median", return_forecast=True)
v = d["lam_back"].loc[:"2026-08-26"].dropna()
span = (v.index[-1964], v.index[-1])
sk_rows = []
for mk, s in (("NIFTY", (d["IV2"] - Fi).loc[span[0]:span[1]]), ("US", (du["IV2"] - Fu).loc["2008-01-01":])):
    s = s.loc[s.first_valid_index():]
    bs = F.bootstrap_series(s.to_numpy(dtype=float), (0.02,), L=63, B=2000, seed=31)
    sk_rows.append(dict(market=mk, series="ex-ante pi_hat (logs median)", skew=bs["est"]["skew"], se=bs["se"]["skew"],
                        lo=bs["lo"]["skew"], hi=bs["hi"]["skew"]))
sk = pd.DataFrame(sk_rows)
lines += ["## Skewness of the ex-ante premium estimate (block bootstrap, L = 63)", "", sk.round(3).to_string(index=False), ""]

mom.to_csv(RES / f"13_two_factor_moments{TAG}.csv", index=False)
par.to_csv(RES / f"13_two_factor_params{TAG}.csv", index=False)
sk.to_csv(RES / f"13_exante_skew{TAG}.csv", index=False)
(RES / f"13_two_factor{TAG}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
