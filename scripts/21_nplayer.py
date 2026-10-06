#!/usr/bin/env python3
"""21 - The N-player game: exact symmetric Nash equilibrium and its convergence to the mean-field limit.

N sellers, dX^i = v^i dt + sigma dW^i + sigma0 dW^0, premium lam = a - b m with m = (1/N) sum_j X^j.
Seller i maximises E int e^{-rho t}[(a - b m) X^i - chi/2 (X^i)^2 - eta/2 (v^i)^2] dt and internalises its
own price impact b/N. With n_i the mean of the others, the symmetric Markov Nash equilibrium is
    v^i = (2Q x_i + R n_i + S)/eta,
where (Q, R, P) solve (eps = 1/N)
    E1: rho Q = 2Q^2/eta + eps/(1-eps) R^2/eta - chi/2 - b eps
    E2: rho R = 2QR/eta + eps/(1-eps) R P/eta + (2Q + (1-2eps)/(1-eps) R) R/eta - b (1-eps)
    E3: rho P = R^2/eta + 2 (2Q + (1-2eps)/(1-eps) R) P/eta
and (S, E) solve two linear equations. At eps = 0 the system is triangular with diagonal
(rho + 2 kappa_ind, rho + 2 kappa, rho + 2 kappa) > 0, so the implicit function theorem gives an analytic
branch through the mean-field solution (q0, A, P_inf).

Checks: (1) Nash property by Kleinman policy iteration on (x, n, 1) against the others' strategy;
(2) O(1/N) convergence of all coefficients, speeds and the stationary premium; (3) the market-power
markup against the static Cournot value a b^2 / (N (chi+b)^2); (4) coupling rate E|m_t - Y_t|^2 against
sigma^2/(2 kappa N) by simulation; (5) closed-loop spectrum and existence for all N >= 2 on a grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def mfg(eta, rho, chi, a, b):
    kind = (-rho + np.sqrt(rho ** 2 + 4 * chi / eta)) / 2
    kap = (-rho + np.sqrt(rho ** 2 + 4 * (chi + b) / eta)) / 2
    q0 = -eta * kind / 2
    A = eta * (kind - kap)
    B = a / (rho + kap)
    P = A ** 2 / (eta * (rho + 2 * kap))
    return dict(q0=q0, A=A, B=B, P=P, kind=kind, kap=kap, Ybar=a / (chi + b), lbar=a * chi / (chi + b))


def nplayer(N, eta, rho, chi, a, b, x0=None):
    eps = 1.0 / N
    w, r2 = eps / (1 - eps), (1 - 2 * eps) / (1 - eps)

    def F(z):
        Q, R, P = z
        return [2 * Q ** 2 / eta + w * R ** 2 / eta - chi / 2 - b * eps - rho * Q,
                2 * Q * R / eta + w * R * P / eta + (2 * Q + r2 * R) * R / eta - b * (1 - eps) - rho * R,
                R ** 2 / eta + 2 * (2 * Q + r2 * R) * P / eta - rho * P]
    m = mfg(eta, rho, chi, a, b)
    z0 = x0 if x0 is not None else [m["q0"], m["A"], m["P"]]
    Q, R, P = fsolve(F, z0, xtol=1e-15)
    res = np.max(np.abs(F([Q, R, P])))
    # linear equations for S, E
    M = np.array([[rho - 2 * Q / eta - R / eta, -w * R / eta],
                  [-(R + P) / eta, rho - (2 * Q + r2 * R) / eta]])
    S, E = np.linalg.solve(M, [a, 0.0])
    k = (2 * Q - w * R) / eta          # own-state gain in the (x_j, m) form
    l = R / ((1 - eps) * eta)          # mean gain: R N / ((N-1) eta)
    h = S / eta
    kapN, kindN = -(k + l), -k
    mbar = h / kapN
    return dict(Q=Q, R=R, P=P, S=S, E=E, k=k, l=l, h=h, kapN=kapN, kindN=kindN, mbar=mbar,
                lbar=a - b * mbar, residual=res)


def best_response(N, eta, rho, chi, a, b, sol):
    """Player i's optimal feedback on (x, n, 1) when the others use the symmetric strategy (Kleinman)."""
    k, l, h = sol["k"], sol["l"], sol["h"]
    eps = 1.0 / N
    Fm = np.zeros((3, 3))
    Fm[1, 0], Fm[1, 1], Fm[1, 2] = l * eps, k + l * (1 - eps), h
    G = np.array([[1.0], [0.0], [0.0]])
    Qr = np.zeros((3, 3))
    Qr[0, 0] = -(chi / 2 + b * eps)
    Qr[0, 1] = Qr[1, 0] = -b * (1 - eps) / 2
    Qr[0, 2] = Qr[2, 0] = a / 2
    Fd = Fm - rho / 2 * np.eye(3)
    K = np.array([[-1.0, 0.0, 0.0]])
    for _ in range(500):
        P = solve_continuous_lyapunov((Fd + G @ K).T, -(-Qr + eta / 2 * K.T @ K))
        Kn = -(2 / eta) * (G.T @ P)
        if np.max(np.abs(Kn - K)) < 1e-15:
            K = Kn
            break
        K = Kn
    return K[0]                         # (K_x, K_n, K_1)


out = ["# 21 - N-player Nash equilibrium and convergence", ""]
p0 = dict(eta=1.0, rho=0.10, chi=0.04, a=0.05, b=0.05)
m0 = mfg(**p0)

# (1) Nash verification and (2) rates at the Table-1 parameters
rows = []
for N in (2, 3, 5, 10, 30, 100, 300, 1000, 3000, 10000):
    s = nplayer(N, **p0)
    K = best_response(N, **p0, sol=s)
    nash_err = np.max(np.abs(K - np.array([2 * s["Q"], s["R"], s["S"]]) / p0["eta"]))
    rows.append(dict(N=N, residual=s["residual"], nash_error=nash_err, kindN=s["kindN"], kapN=s["kapN"],
                     lbarN=s["lbar"], N_dQ=N * (s["Q"] - m0["q0"]), N_dR=N * (s["R"] - m0["A"]), N_dS=N * (s["S"] - m0["B"]),
                     N_dkap=N * (s["kapN"] - m0["kap"]), N_dlbar=N * (s["lbar"] - m0["lbar"])))
T = pd.DataFrame(rows)
cournot = p0["a"] * p0["b"] ** 2 / (p0["chi"] + p0["b"]) ** 2
out += ["## Table-1 parameters: Nash verification and N x (N-player - mean field)", T.to_string(index=False), "",
        f"- mean-field: kappa_ind {m0['kind']:.6f}, kappa {m0['kap']:.6f}, lbar {m0['lbar']:.6f}",
        f"- static Cournot markup coefficient a b^2/(chi+b)^2 = {cournot:.6f} (compare N_dlbar)", ""]

# (5) existence, stability and Nash on a random grid, all N >= 2
rng = np.random.default_rng(21)
worst, fails = 0.0, 0
for _ in range(150):
    pp = dict(eta=10 ** rng.uniform(-0.7, 1), rho=10 ** rng.uniform(-2, -0.5), chi=10 ** rng.uniform(-2, -0.3),
              a=10 ** rng.uniform(-2, 0), b=10 ** rng.uniform(-2, -0.3))
    prev = None
    for N in list(range(2000, 1, -1))[::-1][:0] or [10000, 3000, 1000, 300, 100, 30, 10, 5, 3, 2]:
        s = nplayer(N, **pp, x0=prev)
        prev = [s["Q"], s["R"], s["P"]]
        K = best_response(N, **pp, sol=s)
        err = np.max(np.abs(K - np.array([2 * s["Q"], s["R"], s["S"]]) / pp["eta"]))
        worst = max(worst, err, s["residual"])
        fails += int(not (s["kapN"] > 0 and s["kindN"] > 0 and err < 1e-8))
out += [f"## Random grid: 150 parameter sets x N in {{2,...,10000}} (continuation from large N)",
        f"- stable equilibria with kappa_N > 0 and kappa_ind,N > 0 that are verified Nash: {1500 - fails} of 1500",
        f"- max residual / Nash error: {worst:.1e}", ""]

# (4) coupling rate by simulation: same common noise, MFG aggregate vs empirical mean
sig, sig0, dt, T_end, reps = 0.2, 0.15, 0.02, 400.0, 20
crow = []
for N in (10, 30, 100, 300):
    s = nplayer(N, **p0)
    msq = []
    for rep in range(reps):
        r = np.random.default_rng(1000 * N + rep)
        x = np.full(N, m0["Ybar"])
        Y = m0["Ybar"]
        acc, cnt = 0.0, 0
        for t in range(int(T_end / dt)):
            dW0 = np.sqrt(dt) * r.standard_normal()
            dWi = np.sqrt(dt) * r.standard_normal(N)
            m = x.mean()
            x = x + (s["k"] * x + s["l"] * m + s["h"]) * dt + sig * dWi + sig0 * dW0
            Y = Y - m0["kap"] * (Y - m0["Ybar"]) * dt + sig0 * dW0
            if t * dt > 50:
                acc += (x.mean() - Y) ** 2
                cnt += 1
        msq.append(acc / cnt)
    crow.append(dict(N=N, E_sq_gap=np.mean(msq), se=np.std(msq, ddof=1) / np.sqrt(reps),
                     theory=sig ** 2 / (2 * m0["kap"] * N), ratio=np.mean(msq) / (sig ** 2 / (2 * m0["kap"] * N))))
Cp = pd.DataFrame(crow)
slope = np.polyfit(np.log(Cp.N), np.log(Cp.E_sq_gap), 1)[0]
out += ["## Coupling: E|m_t - Y_t|^2 (stationary, same common noise) against sigma^2/(2 kappa N)",
        Cp.to_string(index=False), f"- log-log slope in N: {slope:.3f} (theory -1)", ""]

T.to_csv(RES / "21_nplayer_rates.csv", index=False)
Cp.to_csv(RES / "21_nplayer_coupling.csv", index=False)
(RES / "21_nplayer.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
