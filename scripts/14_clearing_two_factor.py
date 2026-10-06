#!/usr/bin/env python3
"""14 - Does the clearing relation follow the two-factor model?

Theory (Section 7): in (lam, ytil) coordinates, ytil = b (Y - Ybar), the equilibrium is linear with
    d lam  = [ -(kappa_a + beta) lam + (kappa - kappa_a - beta) ytil ] dt + r dW^a - dW^0
    d ytil = [  beta lam - (kappa - beta) ytil ] dt + dW^0,          beta = b C / eta,
so a high premium raises future deployment (beta > 0: the equilibrium supply response) and, when
demand is the faster state (kappa_a > kappa - beta), high deployment lowers the future premium.

Tests on weekly NIFTY data (premium proxy, detrended log short open interest), 2020-2026:
  V1 weekly VAR(1): signs of the cross effects; scale-free comparison of Phi_11, Phi_22 and
     Phi_12 * Phi_21 with the values implied by the parameters fitted in script 13;
  V2 horizon profile of the correlation of h-week changes, h = 1, 2, 4, 8, 13, against the fitted
     model (the fit used only h = 1 and levels);
  V3 slopes: the 1-week change slope (identifies -b in proxy units as h -> 0) against the level
     slope (biased upward by demand);
  V4 variance decomposition and demand pass-through implied by the fitted parameters.
Standard errors: moving-block bootstrap of weeks, L = 8 (L = max(8, 2h) for horizon h), B = 2000.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import inference as F, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402
from vmfg import twofactor as TF  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
D = 1.0 / 52.0
B = 2000
RHO0 = 0.05
rng = np.random.default_rng(14)

india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
wk = d[["lam_back"]].join(poi[["log_short_detr"]], how="inner").resample("W-FRI").mean().dropna()
import os as _os
OI_END = _os.environ.get("VMFG_OI_END")
TAG = _os.environ.get("VMFG_TAG", "")
if OI_END:
    wk = wk.loc[:OI_END]
W = wk.to_numpy()                  # columns: lam, Y
n = len(W)

par = pd.read_csv(RES / f"13_two_factor_params{TAG}.csv").set_index("parameter")["estimate"]
kap, kap_a, s_sh, r = (par["kappa (1/yr)"], par["kappa_a (1/yr)"], par["s = b/(chi+b)"], par["r = sigma_a/(b sigma0)"])
beta = s_sh * kap * (kap + RHO0) / (RHO0 + kap + kap_a)
M_la = np.array([[-(kap_a + beta), kap - kap_a - beta], [beta, -(kap - beta)]])   # (lam, ytil)
M_ya = np.array([[-kap, beta], [0.0, -kap_a]])                                    # (ytil, alpha)
Q_ya = np.diag([1.0, r * r])
S_ya = solve_continuous_lyapunov(M_ya, -Q_ya)
T = np.array([[-1.0, 1.0], [1.0, 0.0]])        # (lam, ytil) = T (ytil, alpha)
S_la = T @ S_ya @ T.T


def var1(X):
    Y1, X0 = X[1:], np.column_stack([np.ones(len(X) - 1), X[:-1]])
    coef, *_ = np.linalg.lstsq(X0, Y1, rcond=None)
    return coef[1:].T                      # Phi: rows = equation (lam, Y), cols = regressor (lam, Y)


def pair_blocks(m, L):
    return F.block_indices(m, L, rng)


# ---------------- V1: VAR(1)
Phi = var1(W)
reps = []
pairs0, pairs1 = W[:-1], W[1:]
for _ in range(B):
    ib = pair_blocks(n - 1, 8)
    X0 = np.column_stack([np.ones(len(ib)), pairs0[ib]])
    coef, *_ = np.linalg.lstsq(X0, pairs1[ib], rcond=None)
    reps.append(coef[1:].T)
reps = np.array(reps)
se_Phi = reps.std(axis=0, ddof=1)
prod = reps[:, 0, 1] * reps[:, 1, 0]
MM = TF.moments(kap, kap_a, beta, r, averaged=True, horizons=(1, 2, 4, 8, 13))
Phi_model = MM["Phi"]
v1 = pd.DataFrame([
    dict(quantity="Phi_lam,lam (premium persistence)", data=Phi[0, 0], se=se_Phi[0, 0], model=Phi_model[0, 0]),
    dict(quantity="Phi_Y,Y (supply persistence)", data=Phi[1, 1], se=se_Phi[1, 1], model=Phi_model[1, 1]),
    dict(quantity="Phi_Y,lam (premium -> future supply)", data=Phi[1, 0], se=se_Phi[1, 0], model=np.nan),
    dict(quantity="Phi_lam,Y (supply -> future premium)", data=Phi[0, 1], se=se_Phi[0, 1], model=np.nan),
    dict(quantity="Phi_lam,Y * Phi_Y,lam (scale-free)", data=Phi[0, 1] * Phi[1, 0], se=prod.std(ddof=1),
         model=Phi_model[0, 1] * Phi_model[1, 0]),
])
v1["z_vs_model"] = (v1["data"] - v1["model"]) / v1["se"]
v1["t_vs_zero"] = v1["data"] / v1["se"]
signs = dict(model_premium_to_supply=np.sign(Phi_model[1, 0]), model_supply_to_premium=np.sign(Phi_model[0, 1]),
             p_premium_to_supply_le0=float(np.mean(reps[:, 1, 0] <= 0)),
             p_supply_to_premium_ge0=float(np.mean(reps[:, 0, 1] >= 0)))


# ---------------- V2: horizon profile of correlations of h-week changes
def model_corr_changes(h_weeks):
    return float(MM[f"corr_chg_{h_weeks}"])


l = np.array([-1.0, 1.0]); e1 = np.array([1.0, 0.0])
model_levels = float(MM["corr_lev"])
v2_rows = []
for h in (1, 2, 4, 8, 13):
    dl = W[h:, 0] - W[:-h, 0]
    dy = W[h:, 1] - W[:-h, 1]
    c = np.corrcoef(dl, dy)[0, 1]
    Lh = max(8, 2 * h)
    cb = [np.corrcoef(dl[ib], dy[ib])[0, 1] for ib in (pair_blocks(len(dl), Lh) for _ in range(B))]
    se = float(np.std(cb, ddof=1))
    mc = model_corr_changes(h)
    v2_rows.append(dict(horizon_weeks=h, data=c, se=se, model=mc, z=(c - mc) / se, used_in_fit=(h == 1)))
cl = np.corrcoef(W[:, 0], W[:, 1])[0, 1]
clb = [np.corrcoef(W[ib, 0], W[ib, 1])[0, 1] for ib in (pair_blocks(n, 8) for _ in range(B))]
v2_rows.append(dict(horizon_weeks=np.inf, data=cl, se=float(np.std(clb, ddof=1)), model=model_levels,
                    z=(cl - model_levels) / float(np.std(clb, ddof=1)), used_in_fit=True))
v2 = pd.DataFrame(v2_rows)

# ---------------- V3: slopes
dl1, dy1 = np.diff(W[:, 0]), np.diff(W[:, 1])
slope_ch = np.cov(dl1, dy1)[0, 1] / np.var(dy1, ddof=1)
slope_lv = np.cov(W[:, 0], W[:, 1])[0, 1] / np.var(W[:, 1], ddof=1)
bs = []
for _ in range(B):
    ib = pair_blocks(n - 1, 8)
    a0, a1 = W[:-1][ib], W[1:][ib]
    dl, dy = a1[:, 0] - a0[:, 0], a1[:, 1] - a0[:, 1]
    ic = pair_blocks(n, 8)
    s_ch = np.cov(dl, dy)[0, 1] / np.var(dy, ddof=1)
    s_lv = np.cov(W[ic, 0], W[ic, 1])[0, 1] / np.var(W[ic, 1], ddof=1)
    bs.append((s_ch, s_lv, s_lv - s_ch))
bs = np.array(bs)
v3 = pd.DataFrame([
    dict(quantity="1-week change slope (estimates -b, proxy units)", est=slope_ch, se=bs[:, 0].std(ddof=1)),
    dict(quantity="level slope (-b + demand bias)", est=slope_lv, se=bs[:, 1].std(ddof=1)),
    dict(quantity="level slope minus change slope (demand bias > 0)", est=slope_lv - slope_ch, se=bs[:, 2].std(ddof=1)),
])
v3["t"] = v3["est"] / v3["se"]

# ---------------- V4: variance decomposition and pass-through (fitted parameters, scaled units)
V_sup = 1.0 / (2 * kap)                                 # b^2 sigma0^2 / (2 kappa) with b sigma0 = 1
V_lam = float(l @ S_ya @ l)
share_demand = 1 - V_sup / V_lam
pass_through = 1 - beta / kap
t_peak = np.log(kap / kap_a) / (kap - kap_a) if kap != kap_a else 1 / kap
v4 = pd.DataFrame([dict(quantity="demand share of premium variance", value=share_demand),
                   dict(quantity="integrated pass-through of a demand shock with persistence kappa_a, 1 - bg/kappa", value=pass_through),
                   dict(quantity="pass-through of a permanent demand shift, chi/(chi+b) = 1 - s", value=1 - s_sh),
                   dict(quantity="peak of the supply response to a demand shock (weeks)", value=52 * t_peak)])

for nm, df in (("V1_var", v1), ("V2_horizon", v2), ("V3_slopes", v3), ("V4_decomposition", v4)):
    df.to_csv(RES / f"14_{nm}{TAG}.csv", index=False)
pd.set_option("display.width", 200)
out = ["# 14 - Does clearing follow the two-factor model?", "",
       f"fitted parameters: kappa={kap:.3f}, kappa_a={kap_a:.3f}, s={s_sh:.3f}, r={r:.3f}, beta={beta:.4f}; weeks={n}", "",
       "## V1 weekly VAR(1)", v1.round(4).to_string(index=False), "", f"signs: {signs}", "",
       "## V2 horizon profile of change correlations", v2.round(4).to_string(index=False), "",
       "## V3 slopes", v3.round(5).to_string(index=False), "", "## V4 implied decomposition", v4.round(4).to_string(index=False)]
(RES / f"14_clearing_two_factor{TAG}.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
