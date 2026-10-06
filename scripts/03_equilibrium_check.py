#!/usr/bin/env python3
"""03 - Independent confirmation of Theorems 5.3-5.4 and Proposition 6.1.

For the seven Table 1 rows and 300 random parameter draws:
  (a) MFG fixed point by Riccati best response (no shadow potential used):
      bracketing root-finder for kappa, exact affine solve for Ybar;
  (b) Lucas-Prescott planner aggregate (its own Riccati equation);
  (c) exact discounted payoff of the equilibrium feedback (Lyapunov equation)
      versus V(x, Y) = q0 x^2 + x u(Y) + w(Y) at random points;
  (d) optimality: 2000 random perturbations of the feedback never improve
      the payoff, and a local optimiser started away from K* returns to it;
  (e) the rejected root A_plus gives an explosive aggregate (A_plus + 2 q0 > 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import lq_check as L, model as M  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)

draws = [M.Params(**{**p.to_dict(), "sigma": 0.2, "sigma0": 0.15}) for p in M.TABLE1_ROWS]
for _ in range(300):
    draws.append(M.Params(a=10 ** rng.uniform(-2, 0), b=10 ** rng.uniform(-3, 0), chi=10 ** rng.uniform(-3, 0),
                          eta=10 ** rng.uniform(-1, 1.5), rho=10 ** rng.uniform(-2.5, -0.5),
                          sigma=rng.uniform(0.05, 0.5), sigma0=rng.uniform(0.02, 0.4)))

err_fp, err_pl, err_val, worst_gain, aplus_ok, opt_back = [], [], [], -np.inf, True, []
for j, p in enumerate(draws):
    fp = L.equilibrium_fixed_point(p)
    err_fp.append(max(abs(fp["kappa"] - M.kappa(p)) / M.kappa(p), abs(fp["Ybar"] - M.Ybar(p)) / M.Ybar(p),
                      abs(fp["A"] - M.A(p)) / abs(M.A(p)), abs(fp["B"] - M.B(p)) / M.B(p),
                      abs(fp["q0"] - M.q0(p)) / abs(M.q0(p))))
    kp, yp = L.planner_aggregate(p)
    err_pl.append(max(abs(kp - M.kappa(p)) / M.kappa(p), abs(yp - M.Ybar(p)) / M.Ybar(p)))
    K = np.array(M.feedback(p))
    yb, sd = M.Ybar(p), M.s_star(p)
    for _ in range(3):
        x0, y0 = yb + rng.normal() * (1 + sd), yb + rng.normal() * sd
        J, V = L.policy_value(p, K, x0, y0), float(M.value(p, x0, y0))
        err_val.append(abs(J - V) / max(1.0, abs(V)))
    am, ap = M.A_roots(p)
    aplus_ok &= (ap + 2 * M.q0(p) > 0) and (am + 2 * M.q0(p) < 0)
    if j < 40:  # optimality probes (costlier)
        x0, y0 = yb + 0.3, yb - 0.5 * sd
        J0 = L.policy_value(p, K, x0, y0)
        scale = np.abs(K) + 1e-3
        for _ in range(50):
            dK = rng.normal(size=3) * scale * 10 ** rng.uniform(-4, -1)
            worst_gain = max(worst_gain, (L.policy_value(p, K + dK, x0, y0) - J0) / max(1.0, abs(J0)))
        start = K + rng.normal(size=3) * 0.2 * scale
        res = minimize(lambda kk: -L.policy_value(p, kk, x0, y0), start, method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        opt_back.append(np.max(np.abs(res.x - K) / scale))

lines = ["# 03 - Independent equilibrium checks", "",
         f"Parameter sets: {len(draws)} (7 from Table 1 + 300 random, log-uniform over wide ranges).", "",
         "| check | worst case | verdict |", "|---|---|---|",
         f"| (a) Riccati best response + fixed point = closed forms (relative error) | {max(err_fp):.2e} | "
         f"{'PASS' if max(err_fp) < 1e-9 else 'FAIL'} |",
         f"| (b) Lucas-Prescott planner aggregate = equilibrium aggregate | {max(err_pl):.2e} | "
         f"{'PASS' if max(err_pl) < 1e-9 else 'FAIL'} |",
         f"| (c) exact payoff of v* (Lyapunov) = V(x,Y) incl. w from (9) | {max(err_val):.2e} | "
         f"{'PASS' if max(err_val) < 1e-8 else 'FAIL'} |",
         f"| (d1) best payoff gain from 2000 random feedback perturbations | {worst_gain:.2e} | "
         f"{'PASS' if worst_gain <= 1e-12 else 'FAIL'} |",
         f"| (d2) Nelder-Mead from a perturbed start returns to K* (scaled distance) | {max(opt_back):.2e} | "
         f"{'PASS' if max(opt_back) < 1e-4 else 'CHECK'} |",
         f"| (e) A_minus + 2 q0 < 0 < A_plus + 2 q0 in every draw | - | {'PASS' if aplus_ok else 'FAIL'} |", "",
         "Interpretation: the equilibrium of Theorems 5.3-5.4 is reproduced by the textbook MFG fixed point, "
         "and it coincides with the solution of the planner problem "
         "max E int e^{-rho t}[aY - ((b+chi)/2)Y^2 - (eta/2)v^2] dt, dY = v dt + sigma0 dW0 "
         "(Lucas and Prescott, 1971; the potential-game structure of Lasry-Lions). "
         "Section 4's shadow potential S equals that planner value minus q0 Y^2 (01_symbolic_checks.md)."]
(OUT / "03_equilibrium_check.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
