#!/usr/bin/env python3
"""20 - Control-theoretic results: numerical verification.

(a) Mean-level best response. Against a posited aggregate law dY = -khat (Y - Yhat) dt + sigma0 dW^0, the
    individual's optimal linear feedback is computed by Kleinman policy iteration on (X, Y, 1), with no
    ansatz; its stationary mean position E[X] is compared with (a - b Yhat)/chi, which does not depend on khat.
(b) Adaptation. Iterating the exact best response on the mean level converges iff b < chi; the damped
    iteration converges iff 0 < gamma < 2 chi/(chi + b).
(c) Disturbance rejection. The frequency response from demand to premium of the two-factor loop, computed
    from (i w I - M)^{-1}, equals S(iw)/(iw + kappa_a) with S(p) = 1 - b g/(p + kappa), |S| rising from
    1 - b g/kappa at w = 0 to 1 as w -> infinity.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_lyapunov

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402

rng = np.random.default_rng(20)


def best_response_mean(eta, rho, chi, a, b, khat, Yhat):
    F = np.zeros((3, 3)); F[1, 1], F[1, 2] = -khat, khat * Yhat
    G = np.array([[1.0], [0.0], [0.0]])
    Q = np.zeros((3, 3)); Q[0, 0] = -chi / 2; Q[0, 1] = Q[1, 0] = -b / 2; Q[0, 2] = Q[2, 0] = a / 2
    Fd = F - rho / 2 * np.eye(3)
    K = np.array([[-1.0, 0.0, 0.0]])
    for _ in range(300):
        P = solve_continuous_lyapunov((Fd + G @ K).T, -(-Q + eta / 2 * K.T @ K))
        Kn = -(2 / eta) * (G.T @ P)
        if np.max(np.abs(Kn - K)) < 1e-15:
            K = Kn
            break
        K = Kn
    kX, kY, k1 = K[0]
    return -(kY * Yhat + k1) / kX          # stationary E[X] when E[Y] = Yhat


err_a = 0.0
for _ in range(200):
    eta, rho, chi, a, b = 10 ** rng.uniform(-0.7, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3), \
        10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, -0.3)
    khat, Yhat = 10 ** rng.uniform(-1.5, 1.5), rng.uniform(-2, 2)
    err_a = max(err_a, abs(best_response_mean(eta, rho, chi, a, b, khat, Yhat) - (a - b * Yhat) / chi) / (1 + abs((a - b * Yhat) / chi)))
print(f"(a) mean-level best response = (a - b Yhat)/chi, independent of khat: max rel. error {err_a:.1e} (200 sets)")

ok_b = 0
for _ in range(200):
    chi, b, a = 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0), 10 ** rng.uniform(-2, 0)
    gam = rng.uniform(0.05, 1.5)
    Ybar = a / (chi + b)
    if min(abs(b / chi - 1), abs(abs(1 - gam * (1 + b / chi)) - 1)) < 1e-3:
        ok_b += 1                         # knife-edge case: neither converges nor diverges
        continue
    def run(slope):                       # error recursion of a linear iteration, run on its own
        e = 1.0
        for _ in range(20000):
            e *= slope
            if abs(e) > 1e12 or abs(e) < 1e-300:
                break
        return e
    e = run(-(b / chi))                   # undamped best-response iteration
    ed = run(1 - gam * (1 + b / chi))     # damped iteration
    # direct iteration of the map, to confirm the error recursion used above
    Y = Ybar + 1.0
    for _ in range(50):
        Y = (a - b * Y) / chi
    assert abs((Y - Ybar) - (-(b / chi)) ** 50) < 1e-6 * max(1.0, abs((b / chi) ** 50))
    conv, convd = abs(e) < 1e-6, abs(ed) < 1e-6
    pred, predd = b < chi, 0 < gam < 2 * chi / (chi + b)
    ok_b += int(conv == pred and convd == predd)
print(f"(b) convergence iff b < chi, damped iff gamma < 2chi/(chi+b): {ok_b} of 200 cases agree")

err_c, mono = 0.0, 0
for _ in range(200):
    eta, rho, chi, b, ka = (10 ** rng.uniform(-0.7, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3),
                            10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3))
    p = TF.closed_forms(eta, rho, chi, b, ka, 1.0)
    k, g = p["kappa"], p["g"]
    M = np.array([[-k, g], [0.0, -ka]])
    ws = np.logspace(-3, 3, 200)
    mags = []
    for w in ws:
        H = np.array([-b, 1.0]) @ np.linalg.solve(1j * w * np.eye(2) - M, np.array([0.0, 1.0]))
        S = 1 - b * g / (1j * w + k)
        err_c = max(err_c, abs(H - S / (1j * w + ka)) / abs(S / (1j * w + ka)))
        mags.append(abs(S))
    S0 = abs(1 - b * g / k)
    mono += int(np.all(np.diff(mags) >= -1e-12) and b * g < k and mags[0] >= S0 - 1e-12 and mags[-1] <= 1 + 1e-12)
print(f"(c) frequency response demand -> premium = S(iw)/(iw+kappa_a): max rel. error {err_c:.1e}; "
      f"|S| increasing from 1 - bg/kappa: {mono} of 200")
