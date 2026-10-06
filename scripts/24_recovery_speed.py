#!/usr/bin/env python3
"""24 - How fast does the loop restore equilibrium after a capacity crossing?

Stationary two-factor equilibrium: E[lam_{t+tau} | lam_t = 0] = lbar (1 - rho(tau)), rho the autocorrelation of lam,
    rho(tau) = w_s e^{-kappa tau} + w_d e^{-kappa_a tau},   w_s + w_d = 1,
    w_d = (V_alpha - b C_ya)(1 - b g/(kappa - kappa_a)) / V_lam,   w_s = 1 - w_d.
Speeds: kappa_0 = w_s kappa + w_d kappa_a (initial), T_bar = w_s/kappa + w_d/kappa_a (mean recovery time),
min(kappa, kappa_a) (asymptotic), tau_half root of rho = 1/2.
One-factor: rho = e^{-kappa tau}; exact expected first-passage time from lam = 0 to lam = lbar:
    E[T] = (sqrt(pi)/kappa) int_0^d e^{u^2} erfc(u) du,   d = lbar sqrt(kappa)/(b sigma0).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq
from scipy.special import erfcx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402

rng = np.random.default_rng(24)


def weights(k, ka, g, b, s0, sa):
    Va = sa ** 2 / (2 * ka)
    Cya = g * Va / (k + ka)
    Vy = s0 ** 2 / (2 * k) + g * Cya / k
    V = Va - 2 * b * Cya + b ** 2 * Vy
    wd = (Va - b * Cya) * (1 - b * g / (k - ka)) / V
    return 1 - wd, wd, V


dev, dev0, devT, neg = 0.0, 0.0, 0.0, 0
for _ in range(400):
    eta, rho, chi, b, ka = 10 ** rng.uniform(-0.7, 1), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3), \
        10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3)
    s0, sa = 10 ** rng.uniform(-1.5, 0), 10 ** rng.uniform(-1.5, 0.5)
    cf = TF.closed_forms(eta, rho, chi, b, ka, 1.0)
    k, g = cf["kappa"], cf["g"]
    if abs(k - ka) < 1e-6:
        continue
    ws, wd, V = weights(k, ka, g, b, s0, sa)
    M = np.array([[-k, g], [0.0, -ka]]); S = solve_continuous_lyapunov(M, -np.diag([s0 ** 2, sa ** 2])); l = np.array([-b, 1.0])
    for t in np.linspace(0, 8 / min(k, ka), 25):
        dev = max(dev, abs(l @ expm(M * t) @ S @ l / (l @ S @ l) - (ws * np.exp(-k * t) + wd * np.exp(-ka * t))))
    dev0 = max(dev0, abs((ws * k + wd * ka) - (sa ** 2 + b ** 2 * s0 ** 2) / (2 * V)) / ((sa ** 2 + b ** 2 * s0 ** 2) / (2 * V)))
    Tq = quad(lambda t: ws * np.exp(-k * t) + wd * np.exp(-ka * t), 0, np.inf, limit=200)[0]
    devT = max(devT, abs(Tq - (ws / k + wd / ka)) / (ws / k + wd / ka))
    slow_w = ws if k < ka else wd
    neg += int(slow_w < 0)
print(f"(a) autocorrelation = w_s e^(-kappa tau) + w_d e^(-kappa_a tau): max deviation {dev:.1e}")
print(f"(b) kappa_0 = w_s kappa + w_d kappa_a: max rel. deviation {dev0:.1e}; T_bar = w_s/kappa + w_d/kappa_a: {devT:.1e}")
print(f"(c) weight of the slower mode negative (recovery overshoots lbar): {neg} of 400 parameter sets")

# (d) one-factor exact first-passage time vs numerical ODE solution and Monte Carlo
k, s0, b, lbar = 0.2541, 0.15, 0.05, 0.022222
sd = b * s0 / np.sqrt(2 * k)                         # stationary s.d. of the premium
d = lbar * np.sqrt(k) / (b * s0)
ET = np.sqrt(np.pi) / k * quad(lambda u: erfcx(u), 0, d)[0]   # e^{u^2} erfc(u) = erfcx(u)
sig_l = b * s0
def first_passage_mc(dt, paths=20000, seed=7):
    """Exact OU transitions plus a Brownian-bridge crossing test between grid points (continuous monitoring)."""
    r = np.random.default_rng(seed)
    a = np.exp(-k * dt)
    sd_step = sig_l * np.sqrt((1 - a ** 2) / (2 * k))
    x = np.zeros(paths); T = np.full(paths, np.nan); alive = np.ones(paths, bool); t = 0.0
    while alive.any() and t < 400:
        idx = np.where(alive)[0]
        xn = lbar + (x[idx] - lbar) * a + sd_step * r.standard_normal(idx.size)
        crossed = xn >= lbar
        both_below = ~crossed
        p_bridge = np.exp(-2 * (lbar - x[idx][both_below]) * (lbar - xn[both_below]) / (sig_l ** 2 * dt))
        crossed[both_below] = r.random(both_below.sum()) < p_bridge
        t += dt
        T[idx[crossed]] = t - dt / 2
        alive[idx[crossed]] = False
        x[idx] = xn
    return np.nanmean(T), np.nanstd(T, ddof=1) / np.sqrt(np.isfinite(T).sum())


mc = {dt: first_passage_mc(dt) for dt in (0.01, 0.002)}
print(f"(d) one-factor first passage 0 -> lbar: formula {ET:.4f}; bridge-corrected Monte Carlo "
      f"{mc[0.01][0]:.4f} ({mc[0.01][1]:.4f}) at dt = 0.01 and {mc[0.002][0]:.4f} ({mc[0.002][1]:.4f}) at dt = 0.002; "
      f"d = {d:.3f}; half-life ln2/kappa = {np.log(2)/k:.3f}; 1/kappa = {1/k:.3f}")

# (e) fitted NIFTY parameters (weakly identified; scaled units, b sigma0 = 1)
par = pd.read_csv(ROOT / "results" / "13_two_factor_params.csv").set_index("parameter")["estimate"]
kf, kaf, sf, rf = par["kappa (1/yr)"], par["kappa_a (1/yr)"], par["s = b/(chi+b)"], par["r = sigma_a/(b sigma0)"]
beta = TF.beta_of(kf, kaf, sf)
ws, wd, V = weights(kf, kaf, beta, 1.0, 1.0, rf)    # scaled: b = 1, sigma0 = 1, g = beta, sigma_a = r
k0 = ws * kf + wd * kaf
Tbar = ws / kf + wd / kaf
th = brentq(lambda t: ws * np.exp(-kf * t) + wd * np.exp(-kaf * t) - 0.5, 1e-6, 10)
# overshoot: when the slower mode has a negative weight
(ksl, wsl), (kf_, wf_) = sorted([(kf, ws), (kaf, wd)])
if wsl < 0:
    tau_c = np.log(-wf_ / wsl) / (kf_ - ksl)
    tau_star = np.log(-kf_ * wf_ / (ksl * wsl)) / (kf_ - ksl)
    over = -(ws * np.exp(-kf * tau_star) + wd * np.exp(-kaf * tau_star))
    # check tau_c against a numerical root and tau_star against a numerical maximum
    tc_num = brentq(lambda t: ws * np.exp(-kf * t) + wd * np.exp(-kaf * t), 1e-6, 5)
    grid = np.linspace(0, 3, 300001)
    ts_num = grid[np.argmin(ws * np.exp(-kf * grid) + wd * np.exp(-kaf * grid))]
    print(f"    overshoot: expected premium regains lbar at {52*tau_c:.2f} weeks (numerical root {52*tc_num:.2f}); "
          f"maximum overshoot {100*over:.2f}% of lbar at {52*tau_star:.2f} weeks (numerical {52*ts_num:.2f})")
print(f"(e) fitted NIFTY (kappa {kf:.1f}, kappa_a {kaf:.1f} per year): w_s = {ws:.3f}, w_d = {wd:.3f}; "
      f"initial speed kappa_0 = {k0:.1f}/yr; mean recovery time {52*Tbar:.1f} weeks; half-life {52*th:.1f} weeks")
pd.DataFrame([dict(w_s=ws, w_d=wd, kappa0=k0, Tbar_weeks=52 * Tbar, half_life_weeks=52 * th, onefactor_ET=ET, onefactor_mc=mc[0.002][0])]
             ).to_csv(ROOT / "results" / "24_recovery.csv", index=False)
