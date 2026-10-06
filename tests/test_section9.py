"""Theorem 9.12 of the revised manuscript (sigma0 = 0): free-boundary solve vs eqs. (42)-(44)."""
import numpy as np
from scipy.optimize import brentq, fsolve


def test_theorem_9_12_band_and_premium():
    chi, rho, sigma, cp, cm, a, b = 0.09, 0.08, 0.15, 0.02, 0.40, 0.08, 0.10
    th = np.sqrt(2 * rho) / sigma
    Vp = lambda u, K1, K2: -(chi / rho) * u + K1 * np.cosh(th * u) + K2 * np.sinh(th * u)
    Vpp = lambda u, K1, K2: -(chi / rho) + th * (K1 * np.sinh(th * u) + K2 * np.cosh(th * u))
    h42 = brentq(lambda h: (chi / rho) * (h - np.tanh(th * h) / th) - (cp + cm) / 2, 1e-9, 1e3)
    d43 = -rho * (cp - cm) / (2 * chi)
    sol = fsolve(lambda z: [Vp(z[0], z[2], z[3]) - cp, Vp(z[1], z[2], z[3]) + cm,
                            Vpp(z[0], z[2], z[3]), Vpp(z[1], z[2], z[3])],
                 [d43 - 1.3 * h42, d43 + 0.7 * h42, 0.0, 0.0], xtol=1e-13)
    h, delta = (sol[1] - sol[0]) / 2, (sol[1] + sol[0]) / 2
    assert abs(h - h42) < 1e-9 and abs(delta - d43) < 1e-9
    lam = chi * (a - b * delta) / (chi + b)
    assert abs(lam - (a * chi + 0.5 * rho * b * (cp - cm)) / (chi + b)) < 1e-12
