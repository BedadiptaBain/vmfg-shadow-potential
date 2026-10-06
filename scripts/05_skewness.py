#!/usr/bin/env python3
"""05 - Sign of the stationary premium skewness under concave clearing (Section 8).

Theorem 8.3 states Skew(lam) = -Skew(Y) > 0.  That identity holds only for an
affine map Y -> lam; Lambda_c is curved at the same order c.  Three independent
computations of Skew(lam), all against both formulas:
  Q  quadrature under the paper's own first-order drift (Table 2 procedure)
  NL non-perturbative equilibrium: policy iteration on the auxiliary problem
     with the truncated curved clearing curve of Assumption 8.1
  MC Monte Carlo of the reduced aggregate SDE with the NL drift
Symbols: c curvature; s* = sigma0/sqrt(2 kappa); theta as in model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import model as M, numerics as N, particles as Pt  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SETS = {
    "P1 (Table 1 row 1)": M.Params(a=0.05, b=0.05, chi=0.04, eta=1.0, rho=0.10, sigma=0.2, sigma0=0.15),
    "P7 (Table 1 row 7)": M.Params(a=0.08, b=0.10, chi=0.1875, eta=0.5, rho=0.08, sigma=0.2, sigma0=0.15),
}
lines = ["# 05 - Premium skewness under concave clearing", "",
         "Paper (Thm 8.3): Skew(lam) = -Skew(Y) > 0 (right-skewed premium).",
         "Corrected: Skew(lam) = -Skew(Y) - 3 c s*/b = -(c s*/b)(3 - theta) < 0 (left-skewed premium).", ""]
fig_data = {}
for name, p in SETS.items():
    cs = [0.2 * p.b, 0.04 * p.b, 0.01 * p.b, 0.002 * p.b]
    lines += [f"## {name}: {p.to_dict()}",
              f"kappa = {M.kappa(p):.4f}, s* = {M.s_star(p):.4f}, theta = {M.skew_lambda_theta(p):.4f}", "",
              "| c | Skew(Y) quad | eq.(23) | Skew(lam) quad | Skew(lam) NL | drift channel only | Prop. 14.2 | quad/Prop. 14.2 |",
              "|---|---|---|---|---|---|---|---|"]
    rec = []
    for c in cs:
        q = N.skewness_first_order_drift(p, c)
        nl = N.skewness_nonlinear(p, c)
        claim, corr = M.skew_lambda_drift_channel_only(p, c), M.skew_lambda_first_order(p, c)
        rec.append((c, q["skew_lam"], nl["skew_lam"], claim, corr))
        lines.append(f"| {c:.5f} | {q['skew_Y']:+.6f} | {M.skew_Y_first_order(p, c):+.6f} | {q['skew_lam']:+.6f} | "
                     f"{nl['skew_lam']:+.6f} | {claim:+.6f} | {corr:+.6f} | {q['skew_lam'] / corr:.4f} |")
    fig_data[name] = rec
    lines.append("")

# Monte Carlo on the reduced aggregate with the non-perturbative drift
p = SETS["P1 (Table 1 row 1)"]
c = 0.01
nl = N.skewness_nonlinear(p, c)
sol = nl["solution"]
drift_fn = lambda yy: np.interp(yy, sol["y"], sol["drift"])
samples = Pt.simulate_aggregate(drift_fn, M.Ybar(p), p.sigma0, n_paths=4000, T=230.0, dt=0.02,
                                burn=30.0, record_every=0.5, seed=11)
lam_fn = M.clearing_curved(p, c, 6 * M.s_star(p))
lam_s = lam_fn(samples)
sk_pooled = Pt.skewness(lam_s.ravel())
groups = np.array_split(np.arange(samples.shape[1]), 40)      # 40 independent groups of paths
sk_groups = [Pt.skewness(lam_s[:, g].ravel()) for g in groups]
se = np.std(sk_groups, ddof=1) / np.sqrt(len(groups))
lines += ["## Monte Carlo (P1, c = 0.01, 4000 paths x 200 time units after burn-in)", "",
          f"- pooled sample Skew(lam) = {sk_pooled:+.4f} +/- {2 * se:.4f} (2 s.e. from 40 independent path groups; "
          "Euler step dt = 0.02 adds a small bias)",
          f"- non-perturbative quadrature: {nl['skew_lam']:+.4f}; first order (Prop. 14.2): "
          f"{M.skew_lambda_first_order(p, c):+.4f}; drift channel only: {M.skew_lambda_drift_channel_only(p, c):+.4f}", ""]

# comparative statics of the corrected premium skewness (Table 3 analogue)
lines += ["## Premium analogue of Table 3 (b = 0.05, chi = 0.04, rho = 0.10, sigma0 = 0.15)", "",
          "| eta | kappa | abs Skew(Y)/c (paper Table 3) | Skew(lam)/c corrected |", "|---|---|---|---|"]
prev = None
mono = True
for eta_ in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    pp = M.Params(a=0.05, b=0.05, chi=0.04, eta=eta_, rho=0.10, sigma0=0.15)
    val = M.skew_lambda_first_order(pp, 1.0)
    if prev is not None and not abs(val) > abs(prev):
        mono = False
    prev = val
    lines.append(f"| {eta_} | {M.kappa(pp):.4f} | {abs(M.skew_Y_first_order(pp, 1.0)):.3f} | {val:+.3f} |")
lines += ["", f"- |Skew(lam)| increasing in eta: {mono}. Qualitative statics (1)-(3) of Remark 8.4 survive for the "
          "premium, but with the opposite sign, a magnitude roughly 3 s*/b per unit c (about 10x the Y-skewness here), "
          "and a new, explicit dependence on b.", ""]

(OUT / "05_skewness.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (name, rec) in zip(axes, fig_data.items()):
        cc = np.array([r[0] for r in rec])
        ax.semilogx(cc, [r[1] / r[0] for r in rec], "o-", label="numerical (first-order drift)")
        ax.semilogx(cc, [r[2] / r[0] for r in rec], "s--", label="numerical (non-perturbative)")
        ax.semilogx(cc, [r[3] / r[0] for r in rec], "^:", label="Theorem 8.3 as stated")
        ax.semilogx(cc, [r[4] / r[0] for r in rec], "k-", lw=1, label="corrected first order")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(name)
        ax.set_xlabel("clearing curvature c")
        ax.set_ylabel("Skew(premium) / c")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "05_skewness_sign.png", dpi=150)
except Exception as exc:  # plotting is optional
    print("figure skipped:", exc)
