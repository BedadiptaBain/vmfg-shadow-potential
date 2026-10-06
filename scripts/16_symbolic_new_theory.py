#!/usr/bin/env python3
"""16 - Symbolic verification of two new results of Section 7.

(1) General linear demand system: exogenous xi in R^m, d xi = -K xi dt + noise (K stable), demand intercept
    a_t = abar + h' xi_t. Claim: u = A Y + c' xi + abar/(rho + kappa), with A as in (14) and
    c = (K' + (rho + kappa) I)^{-1} h. Checked for m = 2 with a general 2x2 K and a general noise covariance.
(2) First-order curved clearing with stochastic demand: clearing a - bY - (c/2) Y^2, u = u0 + c u1 + O(c^2),
    u0 = A Y + C a + B (Theorem 7.3). Claim: with z = Y - Ybar and alpha = a - abar,
        u1 = P z^2 + Q z alpha + R alpha^2 + S z + T alpha + U,
        P = -1/(2(rho+3kappa)),  Q = 2gP/(rho+2kappa+kappa_a),  R = gQ/(rho+kappa+2kappa_a),
        S = -Ybar/(rho+2kappa),  T = gS/(rho+kappa+kappa_a),  U = (sigma0^2 P + sigma_a^2 R - Ybar^2/2)/(rho+kappa),
    where g = C/eta. The order-c equation is checked to vanish identically in (Y, a).
"""
from pathlib import Path

import sympy as sp

RES = Path(__file__).resolve().parents[1] / "results"
out = ["# 16 - Symbolic verification: general demand system and first-order curvature with demand", ""]

eta, rho, b, s0, abar = sp.symbols("eta rho b sigma_0 abar", positive=True)
ki, k = sp.symbols("kappa_ind kappa", positive=True)
chi = eta * ki * (ki + rho)
bb = eta * k * (k + rho) - chi                       # b expressed through kappa (Proposition 6.1)
q0 = -eta * ki / 2
A = eta * (ki - k)
rho_eff = rho - 2 * q0 / eta
Y = sp.symbols("Y", real=True)

# ------------------------------------------------------------------ (1) general linear demand system, m = 2
x1, x2 = sp.symbols("xi1 xi2", real=True)
k11, k12, k21, k22, h1, h2 = sp.symbols("k11 k12 k21 k22 h1 h2", real=True)
s11, s12, s22 = sp.symbols("s11 s12 s22", real=True)   # noise covariance of xi (any)
K = sp.Matrix([[k11, k12], [k21, k22]])
h = sp.Matrix([h1, h2])
cvec = (K.T + (rho + k) * sp.eye(2)).inv() * h
u = A * Y + cvec[0] * x1 + cvec[1] * x2 + abar / (rho + k)
xi = sp.Matrix([x1, x2])
drift_xi = -K * xi
a_t = abar + (h.T * xi)[0]
gen_u = (drift_xi[0] * sp.diff(u, x1) + drift_xi[1] * sp.diff(u, x2)
         + sp.Rational(1, 2) * (s11 * sp.diff(u, x1, 2) + 2 * s12 * sp.diff(u, x1, x2) + s22 * sp.diff(u, x2, 2)))
eq1 = s0 ** 2 / 2 * sp.diff(u, Y, 2) + gen_u + (u + 2 * q0 * Y) / eta * sp.diff(u, Y) - rho_eff * u + a_t - bb * Y
ok1 = sp.simplify(sp.expand(eq1)) == 0
out += [f"- (1) general linear demand system (m = 2, general K and noise): x^1 equation vanishes identically: {ok1}"]

# ------------------------------------------------------------------ (2) first-order curvature with demand
a, ka, sa, c = sp.symbols("a kappa_a sigma_a c", real=True)
C = 1 / (rho + k + ka)
B = ka * abar * C / (rho + k)
g = C / eta
Ybar = abar / (eta * k * (k + rho))                  # = abar/(chi+b)
u0 = A * Y + C * a + B
z, al = Y - Ybar, a - abar
P = -1 / (2 * (rho + 3 * k))
Q = 2 * g * P / (rho + 2 * k + ka)
R = g * Q / (rho + k + 2 * ka)
S = -Ybar / (rho + 2 * k)
T = g * S / (rho + k + ka)
U = (s0 ** 2 * P + sa ** 2 * R - Ybar ** 2 / 2) / (rho + k)
u1 = P * z ** 2 + Q * z * al + R * al ** 2 + S * z + T * al + U
uu = u0 + c * u1
lam_c = a - bb * Y - c / 2 * Y ** 2
eq2 = (s0 ** 2 / 2 * sp.diff(uu, Y, 2) + sa ** 2 / 2 * sp.diff(uu, a, 2) - ka * (a - abar) * sp.diff(uu, a)
       + (uu + 2 * q0 * Y) / eta * sp.diff(uu, Y) - rho_eff * uu + lam_c)
order0 = sp.simplify(sp.expand(eq2.subs(c, 0)))
order1 = sp.simplify(sp.expand(sp.diff(eq2, c).subs(c, 0)))
out += [f"- (2) first-order curvature with demand: order-0 equation vanishes: {order0 == 0}; "
        f"order-c equation vanishes identically in (Y, a): {order1 == 0}",
        "- (2) reduction check: at sigma_a = 0 and alpha = 0, u1 reduces to the one-factor p Y^2 + r Y + s of (22): "
        + str(sp.simplify(sp.expand((u1.subs({a: abar, sa: 0}) -
              (P * Y ** 2 + (2 * k * Ybar * P / (rho + 2 * k)) * Y + (s0 ** 2 * P + k * Ybar * (2 * k * Ybar * P / (rho + 2 * k))) / (rho + k))))) == 0)]
(RES / "16_symbolic_new.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
