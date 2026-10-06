#!/usr/bin/env python3
"""10 - Error analysis of the numerical methods used to verify the theory.

(1) quadrature: grid refinement of the stationary-density integrals;
(2) non-perturbative solver: grid refinement of the upwind policy iteration,
    observed order of convergence p and Richardson extrapolation
    S_R = S_h + (S_h - S_{2h}) / (2^p - 1);
(3) Monte Carlo of the aggregate: statistical s.e. from 40 independent path groups
    and the time-step bias from dt in {0.04, 0.02, 0.01} (weak order 1:
    S_0 ~ 2 S_dt - S_2dt);
(4) finite population: 20 independent replications, mean +/- s.e., z-score vs theory.
Parameters: Table 1 row 1 with sigma = 0.2, sigma0 = 0.15; curvature c = 0.01.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import model as M, numerics as N, particles as Pt  # noqa: E402

RES = ROOT / "results"
p = M.Params(a=0.05, b=0.05, chi=0.04, eta=1.0, rho=0.10, sigma=0.2, sigma0=0.15)
c = 0.01
yb, s_st = M.Ybar(p), M.s_star(p)
lines = ["# 10 - Numerical error analysis", "", f"parameters {p.to_dict()}, c = {c}", ""]

# (1) quadrature refinement
q_rows = []
for n in (3751, 7501, 15001, 30001, 60001):
    q_rows.append((n, N.skewness_first_order_drift(p, c, n=n)["skew_lam"]))
diffs = [abs(q_rows[i][1] - q_rows[-1][1]) for i in range(len(q_rows) - 1)]
lines += ["## (1) Quadrature (first-order drift), Skew(lam)", "", "| grid points | Skew(lam) | |diff to finest| |",
          "|---|---|---|"] + [f"| {n} | {v:.12f} | {abs(v - q_rows[-1][1]):.2e} |" for n, v in q_rows] + [""]

# (2) policy iteration refinement
pi_rows = []
for n in (2001, 4001, 8001, 16001):
    r = N.skewness_nonlinear(p, c, n=n)
    pi_rows.append((n, r["skew_lam"], r["skew_Y"], r["iterations"]))
S = [r[1] for r in pi_rows]
orders = [np.log2(abs(S[i] - S[i + 1]) / abs(S[i + 1] - S[i + 2])) for i in range(len(S) - 2)]
p_obs = float(orders[-1])
S_R = S[-1] + (S[-1] - S[-2]) / (2 ** p_obs - 1)
err_R = abs(S_R - S[-1])
lines += ["## (2) Non-perturbative solver (upwind policy iteration)", "",
          "| grid points | Skew(lam) | Skew(Y) | iterations |", "|---|---|---|---|"] + \
         [f"| {n} | {sl:.8f} | {sy:.8f} | {it} |" for n, sl, sy, it in pi_rows] + \
         ["", f"- observed orders of convergence: {', '.join(f'{o:.2f}' for o in orders)}",
          f"- Richardson extrapolation: Skew(lam) = {S_R:.6f}, error estimate of the finest grid {err_R:.2e}", ""]

# (3) Monte Carlo with s.e. and time-step refinement
sol = N.skewness_nonlinear(p, c, n=8001)["solution"]
drift_fn = lambda yy: np.interp(yy, sol["y"], sol["drift"])
lam_fn = M.clearing_curved(p, c, 6 * s_st)
mc_rows = []
for dt in (0.04, 0.02, 0.01):
    samp = Pt.simulate_aggregate(drift_fn, yb, p.sigma0, n_paths=4000, T=230.0, dt=dt, burn=30.0,
                                 record_every=0.5, seed=int(1000 * dt) + 7)
    lam = lam_fn(samp)
    groups = np.array_split(np.arange(samp.shape[1]), 40)
    gsk = np.array([Pt.skewness(lam[:, g].ravel()) for g in groups])
    mc_rows.append((dt, Pt.skewness(lam.ravel()), gsk.std(ddof=1) / np.sqrt(len(groups))))
S0 = 2 * mc_rows[2][1] - mc_rows[1][1]
se0 = np.sqrt(4 * mc_rows[2][2] ** 2 + mc_rows[1][2] ** 2)
lines += ["## (3) Monte Carlo of the aggregate (4000 paths, 200 time units after burn-in)", "",
          "| dt | Skew(lam) | s.e. |", "|---|---|---|"] + [f"| {dt} | {v:.4f} | {se:.4f} |" for dt, v, se in mc_rows] + \
         ["", f"- Richardson in dt (weak order 1): Skew(lam) = {S0:.4f} (s.e. {se0:.4f}); "
              f"non-perturbative solver: {S_R:.4f}; difference {S0 - S_R:+.4f} = {(S0 - S_R) / se0:+.2f} s.e.",
          f"- first-order formula (23'): {M.skew_lambda_first_order(p, c):.4f}; drift channel only: {M.skew_lambda_drift_channel_only(p, c):+.4f}", ""]

# (4) finite population replications
reps = []
for k in range(20):
    sim = Pt.simulate_particles(p, N=300, T=1000.0, dt=0.02, burn=50.0, record_every=0.1, seed=100 + k)
    kh, mY, vY = Pt.ar1_speed(sim["Y"], sim["dt_rec"])
    Zn, Zx = sim["Z_now"], sim["Z_next"]
    kin = -np.log(np.sum(Zn * Zx) / np.sum(Zn * Zn)) / sim["dt"]
    lamv = p.a - p.b * sim["Y"]
    reps.append(dict(kappa=kh, kappa_ind=kin, mean_lam=lamv.mean(), disp=sim["disp"].mean()))
R = pd.DataFrame(reps)
theory = dict(kappa=M.kappa(p), kappa_ind=M.kappa_ind(p), mean_lam=M.lam_bar(p), disp=M.Sigma_star(p) * (1 - 1 / 300))
lines += ["## (4) Finite population: 20 independent replications (N = 300, T = 1000)", "",
          "| quantity | theory | mean of replications | s.e. | z |", "|---|---|---|---|---|"]
p_rows = []
for key in ("kappa", "kappa_ind", "mean_lam", "disp"):
    m, se = R[key].mean(), R[key].std(ddof=1) / np.sqrt(len(R))
    p_rows.append(dict(quantity=key, theory=theory[key], mean=m, se=se, z=(m - theory[key]) / se))
    lines.append(f"| {key} | {theory[key]:.5f} | {m:.5f} | {se:.5f} | {(m - theory[key]) / se:+.2f} |")
lines += ["", "Note: the AR(1) estimator of kappa has a finite-sample bias of order 1/T; the z-scores include it.", ""]
pd.DataFrame(p_rows).to_csv(RES / "10_particles.csv", index=False)
pd.DataFrame(dict(dt=[r[0] for r in mc_rows], skew=[r[1] for r in mc_rows], se=[r[2] for r in mc_rows])).to_csv(RES / "10_mc.csv", index=False)
pd.DataFrame(pi_rows, columns=["n", "skew_lam", "skew_Y", "iterations"]).to_csv(RES / "10_pi.csv", index=False)
pd.DataFrame(q_rows, columns=["n", "skew_lam"]).to_csv(RES / "10_quadrature.csv", index=False)
(RES / "10_summary.txt").write_text(f"p_obs={p_obs}\nS_R={S_R}\nerr_R={err_R}\nS0={S0}\nse0={se0}\n")
(RES / "10_numerical_errors.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
