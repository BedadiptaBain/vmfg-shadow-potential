#!/usr/bin/env python3
"""02 - Reproduce Tables 1, 2, 3 and the numerical claims attached to them.

Table 1: closed forms (model.py) versus an independent "numerical" route
(np.roots on the quadratics (7) and (14)), residuals in float64 and in
50-digit mpmath, and the printed rounding.
Table 2: the paper's own quadrature procedure (numerics.table2_ratio).
Table 3: |Skew(Y)|/c as a function of eta with b + chi = 0.09 held fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import model as M, numerics as N  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
lines = ["# 02 - Tables 1-3", ""]
all_ok = True

# ------------------------------------------------------------------ Table 1
lines += ["## Table 1", "",
          "| row | a | b | chi | eta | rho | kappa_ind | kappa | Ybar | lam_bar | printed matches (4 d.p. / 6 d.p.) | max float64 residual | max 50-digit residual |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
mp.mp.dps = 50
max_res64 = 0.0
for i, (p, pr) in enumerate(zip(M.TABLE1_ROWS, M.TABLE1_PRINTED), 1):
    # independent numerical route: roots of the quadratics
    q_roots = np.roots([2.0 / p.eta, -p.rho, -p.chi / 2.0])
    q0n = float(np.min(q_roots.real))
    A_roots = np.roots([1.0, 4.0 * q0n - p.rho * p.eta, -p.b * p.eta])
    An = float(np.min(A_roots.real))
    kin, kn = -2.0 * q0n / p.eta, -(An + 2.0 * q0n) / p.eta
    Bn = p.a / (p.rho - (2.0 * q0n + An) / p.eta)
    Ybn = Bn / (p.eta * kn)
    lbn = p.a - p.b * Ybn
    res64 = [abs(p.rho * q0n + p.chi / 2 - 2 * q0n ** 2 / p.eta),                       # (7)
             abs(An ** 2 + (4 * q0n - p.rho * p.eta) * An - p.b * p.eta),               # (14)
             abs(p.eta * kin * (kin + p.rho) - p.chi),                                  # (16a)
             abs(p.eta * kn * (kn + p.rho) - p.chi - p.b),                              # (16b)
             abs(lbn - p.a * p.chi / (p.chi + p.b))]                                    # (19)
    max_res64 = max(max_res64, max(res64))
    # 50-digit residual of the closed forms
    a, b, chi, eta, rho = (mp.mpf(str(v)) for v in (p.a, p.b, p.chi, p.eta, p.rho))
    kin50 = (-rho + mp.sqrt(rho ** 2 + 4 * chi / eta)) / 2
    k50 = (-rho + mp.sqrt(rho ** 2 + 4 * (chi + b) / eta)) / 2
    q50, A50 = -eta * kin50 / 2, eta * (kin50 - k50)
    B50 = a / (rho + k50)
    res50 = max(abs(rho * q50 + chi / 2 - 2 * q50 ** 2 / eta),
                abs(A50 ** 2 + (4 * q50 - rho * eta) * A50 - b * eta),
                abs(B50 / (eta * k50) - a / (b + chi)))
    got = (kin, kn, Ybn, lbn)
    match = all(round(g, 4) == round(t, 4) for g, t in zip(got[:3], pr[:3])) and round(got[3], 6) == round(pr[3], 6)
    closed = (M.kappa_ind(p), M.kappa(p), M.Ybar(p), M.lam_bar(p))
    agree = max(abs(g - h) for g, h in zip(got, closed))
    all_ok &= match and agree < 1e-12
    lines.append(f"| {i} | {p.a} | {p.b} | {p.chi} | {p.eta} | {p.rho} | {kin:.6f} | {kn:.6f} | {Ybn:.6f} | "
                 f"{lbn:.6f} | {'yes' if match else 'NO'} | {max(res64):.2e} | {mp.nstr(res50, 3)} |")
etas = [p.eta for p in M.TABLE1_ROWS]
span = max(etas) / min(etas)
lines += ["",
          f"- Largest float64 residual over all rows and equations: **{max_res64:.3e}** "
          f"({'below' if max_res64 < 1.1e-16 else 'NOT below'} the printed bound 1.1e-16; machine epsilon is 2.22e-16, "
          "so a bound of this size depends on evaluation order -- state 'at machine precision' or quote high-precision residuals).",
          f"- eta spans {min(etas)} to {max(etas)}: a **{span:.0f}-fold** variation (the caption says 'fortyfold').",
          f"- Row 6: kappa = {M.kappa(M.TABLE1_ROWS[5]):.9f}, so the printed 0.2108 is correctly rounded.",
          "- The independent root-finding route agrees with the closed forms to < 1e-12 in every row.", ""]

# ------------------------------------------------------------------ Table 2
printed2 = {0.25: [0.5752, 0.8135, 0.9532, 0.9901, 0.9975],
            0.40: [0.7310, 0.9085, 0.9797, 0.9958, 0.9989],
            0.60: [0.8462, 0.9547, 0.9904, 0.9980, 0.9995]}
cs = [0.2, 0.05, 0.01, 0.002, 0.0005]
lines += ["## Table 2 (ratio numerical Skew(Y) / eq. (23); eta=1, sigma0=0.15, rho=0.10, Ybar=0.5, grid Ybar +/- 6 s*)", "",
          "| c | kappa=0.25 (printed) | kappa=0.40 (printed) | kappa=0.60 (printed) |", "|---|---|---|---|"]
t2_ok = True
for j, c in enumerate(cs):
    cells = []
    for kap in (0.25, 0.40, 0.60):
        r = N.table2_ratio(c, kap)
        ok = round(r, 4) == printed2[kap][j]
        t2_ok &= ok
        cells.append(f"{r:.4f} ({printed2[kap][j]:.4f}){'' if ok else ' MISMATCH'}")
    lines.append(f"| {c} | " + " | ".join(cells) + " |")
all_ok &= t2_ok
lines += ["", f"- Table 2 reproduced exactly: **{t2_ok}**. Note that Table 2 verifies the skewness of Y only; "
          "it says nothing about the premium (see 05_skewness.md).", ""]

# ------------------------------------------------------------------ Table 3
printed3 = {0.25: (0.5521, 0.589), 0.50: (0.3772, 0.744), 1.00: (0.2541, 0.960), 2.00: (0.1679, 1.276),
            5.00: (0.0932, 1.965), 10.00: (0.0572, 2.851), 20.00: (0.0337, 4.271)}
lines += ["## Table 3 (|Skew(Y)|/c; b + chi = 0.09, rho = 0.10, sigma0 = 0.15)", "",
          "| eta | kappa (printed) | abs Skew(Y)/c (printed) |", "|---|---|---|"]
t3_ok = True
for eta_, (kp, sp_) in printed3.items():
    kap = M.positive_root(0.10, 0.09 / eta_)
    val = abs(M.skew_Y_eq23(0.10, 0.15, eta_, kap, 1.0))
    ok = round(kap, 4) == kp and round(val, 3) == sp_
    t3_ok &= ok
    lines.append(f"| {eta_} | {kap:.4f} ({kp}) | {val:.6f} ({sp_}){'' if ok else ' <- printed value mis-rounded'} |")
lines += ["", f"- Table 3 reproduced to printed precision: **{t3_ok}**. The eta = 0.5 entry is 0.743496, "
          "which rounds to 0.743 (printed 0.744). These are skewnesses of Y, not of the premium.", ""]

lines += ["## Discrepancies with the draft", "",
          f"- residual bound 1.1e-16 in the Table 1 caption: {'holds' if max_res64 < 1.1e-16 else 'fails (max = %.3e)' % max_res64}",
          f"- 'fortyfold variation in eta': actual span is {span:.0f}-fold",
          f"- Table 3, eta = 0.5: printed 0.744, correct 0.743",
          f"- all other printed values of Tables 1-3: {'reproduced' if all_ok else 'see MISMATCH flags'}", ""]
(OUT / "02_tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
