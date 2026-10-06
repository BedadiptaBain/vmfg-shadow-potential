#!/usr/bin/env python3
"""13b - Sensitivity of the two-factor fit to the admissibility bound on s = b/(chi+b).

Under chi > 0 (Assumption 3.2) s lies in the open interval (0, 1). The fit in script 13 allows s <= 1 so
that the boundary chi = 0 can be detected; it reaches s = 1. Here the fit is repeated with s <= s_max
for s_max < 1, to show how much of the fit depends on the boundary. Same targets and weights as script 13.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402

RES = ROOT / "results"
mom = pd.read_csv(RES / "13_two_factor_moments.csv")
target, se = mom["data"].to_numpy(), mom["se"].to_numpy()


def model(p):
    m = TF.moments(p[0], p[1], TF.beta_of(p[0], p[1], p[2]), p[3], averaged=True)
    return np.array([m["k_lam"], m["k_Y"], m["corr_lev"], m["corr_chg_1"]])


rows = []
for s_max in (1.0, 0.99, 0.95, 0.9, 0.75, 0.5, 0.25):
    best = None
    for x0 in ((8.0, 15.0, min(0.9, s_max) * 0.99, 6.0), (3.0, 15.0, 0.5 * s_max, 5.0), (6.0, 12.0, 0.9 * s_max, 6.0),
               (2.5, 20.0, 0.99 * s_max, 10.0), (4.0, 8.0, 0.7 * s_max, 3.0)):
        r = least_squares(lambda p: (model(p) - target) / se, x0, bounds=([0.05, 0.05, 1e-4, 0.01], [60, 200, s_max, 200]))
        if best is None or r.cost < best.cost:
            best = r
    z = (model(best.x) - target) / se
    rows.append(dict(s_max=s_max, chi_over_b_min=(1 - s_max) / s_max, kappa=best.x[0], kappa_a=best.x[1], s=best.x[2],
                     r=best.x[3], sum_z2=float(np.sum(z ** 2)), max_abs_z=float(np.max(np.abs(z)))))
out = pd.DataFrame(rows)
out.to_csv(RES / "13b_fit_boundary.csv", index=False)
print(out.round(4).to_string(index=False))
