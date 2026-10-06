#!/usr/bin/env python3
"""25a - Constant curvature is automatic: against a premium path lam_s = l0 + l1 e^{-g s}, the feedback
v = (2 q0 X + p)/eta, p_t = int_t^inf e^{-rho_eff (s-t)} lam_s ds, satisfies eta v_0 = P_0 := int_0^inf e^{-rho s}(lam_s - chi X_s) ds."""
import numpy as np
from scipy.integrate import quad
rng = np.random.default_rng(25)
worst = 0.0
for _ in range(50):
    eta, rho, chi = 10 ** rng.uniform(-0.5, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3)
    kin = (-rho + np.sqrt(rho ** 2 + 4 * chi / eta)) / 2
    q0 = -eta * kin / 2
    reff = rho + kin
    l0, l1, g, X0 = rng.uniform(-1, 1), rng.uniform(-1, 1), 10 ** rng.uniform(-1.5, 0.5), rng.uniform(-2, 2)
    if abs(kin - g) < 1e-3:
        continue
    lam = lambda s: l0 + l1 * np.exp(-g * s)
    p0 = l0 / reff + l1 / (reff + g)
    X = lambda s: (np.exp(-kin * s) * X0 + (l0 / (eta * reff)) * (1 - np.exp(-kin * s)) / kin
                   + (l1 / (eta * (reff + g))) * (np.exp(-g * s) - np.exp(-kin * s)) / (kin - g))
    P0 = quad(lambda s: np.exp(-rho * s) * (lam(s) - chi * X(s)), 0, np.inf, limit=400)[0]
    worst = max(worst, abs(P0 - (2 * q0 * X0 + p0)) / (1 + abs(P0)))
print(f"(a) first-order condition eta v = P under the constant-curvature feedback: max rel. residual {worst:.1e} (random cases)")
