#!/usr/bin/env python3
"""15 - Verification of the two-factor theory and of the empirical conclusions.

(A) Symbolic: two-factor shadow potential (x^1 equation = d/dY of an auxiliary HJB with exogenous
    demand), the premium SDE, the (lam, ytil) drift matrix, and the cancellation behind the
    deterministic centred law.
(B) Independent equilibrium: each agent's discounted LQ problem on the state (X, Y, a, 1) is solved
    by Kleinman policy iteration (no ansatz for u); the mean-field fixed point in (kappa, g, Ybar) is
    found by root-finding and compared with the closed forms, on random parameter sets.
(C) Numerical identities on random parameter sets: stationary covariances, level covariance,
    positivity condition, short-horizon correlation and slope limits, level slope, variance
    decomposition, integrated pass-through, peak of the supply response.
(D) Simulation of the fitted model at daily frequency, aggregated to weekly means as in the data,
    to measure what temporal aggregation does to the weekly moments and the VAR.
(E) Empirical recomputations: VAR by OLS vs Yule-Walker; Newey-West s.e. of the change slope; the
    VAR with net-seller measures of supply.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import fsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import inference as F, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
out = ["# 15 - Verification of the two-factor theory", ""]

# ======================================================================== (A) symbolic
Y, a = sp.symbols("Y a", real=True)
eta, rho, b, s0, sa, ka, abar, q0 = sp.symbols("eta rho b sigma_0 sigma_a kappa_a abar q_0", real=True)
S = sp.Function("S")(Y, a)
u = S.diff(Y)
L = (s0 ** 2 / 2 * S.diff(Y, 2) + sa ** 2 / 2 * S.diff(a, 2) - ka * (a - abar) * S.diff(a)
     + S.diff(Y) ** 2 / (2 * eta) + 2 * q0 / eta * Y * S.diff(Y) - rho * S + a * Y - b / 2 * Y ** 2)
rho_eff = rho - 2 * q0 / eta
x1 = (s0 ** 2 / 2 * u.diff(Y, 2) + sa ** 2 / 2 * u.diff(a, 2) - ka * (a - abar) * u.diff(a)
      + (u + 2 * q0 * Y) / eta * u.diff(Y) - rho_eff * u + a - b * Y)
okA1 = sp.simplify(L.diff(Y) - x1) == 0
# premium SDE drift and the (lam, ytil) matrix
k, g, al, y = sp.symbols("kappa g alpha y", real=True)
drift_y = -k * y + g * al
drift_al = -ka * al
lam_dev = al - b * y
drift_lam = sp.expand(drift_al - b * drift_y)
claimed = sp.expand(-k * lam_dev + (k - ka - b * g) * al)
okA2 = sp.simplify(drift_lam - claimed) == 0
lt, yt = sp.symbols("lamtil ytil", real=True)            # lamtil = alpha - ytil, ytil = b y
beta = b * g
sub = {al: lt + yt, y: yt / b}
d_yt = sp.expand((b * drift_y).subs(sub))
d_lt = sp.expand(drift_lam.subs(sub))
okA3 = (sp.simplify(d_yt - (beta * lt - (k - beta) * yt)) == 0 and
        sp.simplify(d_lt - (-(ka + beta) * lt + (k - ka - beta) * yt)) == 0)
# centred law: X and Y drifts differ only by -kappa_ind (X - Y), whatever u(Y, a)
X_, kind = sp.symbols("X kappa_ind", real=True)
uf = sp.Function("u")(Y, a)
okA4 = sp.simplify(((2 * q0 * X_ + uf) / eta - (2 * q0 * Y + uf) / eta) - (2 * q0 / eta) * (X_ - Y)) == 0
out += ["## (A) symbolic",
        f"- two-factor shadow potential: x^1 equation = d/dY of the auxiliary HJB: {okA1}",
        f"- premium drift -kappa(lam-lbar) + (kappa-kappa_a-bg) alpha: {okA2}",
        f"- (lam, ytil) drift matrix [[-(kappa_a+beta), kappa-kappa_a-beta],[beta, -(kappa-beta)]]: {okA3}",
        f"- centred law: X - Y drift is -kappa_ind (X - Y) for any u(Y, a): {okA4}", ""]


# ======================================================================== (B) independent equilibrium
def closed_forms(p):
    eta_, rho_, chi_, b_, ka_, ab_ = p
    kind_ = (-rho_ + np.sqrt(rho_ ** 2 + 4 * chi_ / eta_)) / 2
    kap_ = (-rho_ + np.sqrt(rho_ ** 2 + 4 * (chi_ + b_) / eta_)) / 2
    g_ = 1 / (eta_ * (rho_ + kap_ + ka_))
    return kap_, g_, ab_ / (chi_ + b_), kind_


def best_response(p, kh, gh, Yh):
    eta_, rho_, chi_, b_, ka_, ab_ = p
    Fm = np.zeros((4, 4))
    Fm[1, 1], Fm[1, 2], Fm[1, 3] = -kh, gh, kh * Yh - gh * ab_
    Fm[2, 2], Fm[2, 3] = -ka_, ka_ * ab_
    G = np.array([[1.0], [0.0], [0.0], [0.0]])
    Q = np.zeros((4, 4))
    Q[0, 0] = -chi_ / 2
    Q[0, 2] = Q[2, 0] = 0.5
    Q[0, 1] = Q[1, 0] = -b_ / 2
    Fd = Fm - rho_ / 2 * np.eye(4)
    K = np.array([[-1.0, 0.0, 0.0, 0.0]])
    for _ in range(200):
        Fk = Fd + G @ K
        P = solve_continuous_lyapunov(Fk.T, -(-Q + eta_ / 2 * K.T @ K))
        Kn = -(2 / eta_) * (G.T @ P)
        if np.max(np.abs(Kn - K)) < 1e-14:
            K = Kn
            break
        K = Kn
    kX, kY, kA, k1 = K[0]
    kap_new = -(kX + kY)
    g_new = kA
    Y_new = (k1 + g_new * ab_) / kap_new
    return np.array([kap_new, g_new, Y_new])


rng = np.random.default_rng(15)
errs = []
for i in range(60):
    p = (10 ** rng.uniform(-0.7, 1.0), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3),
         10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3), 10 ** rng.uniform(-2, 0))
    kc, gc, Yc, _ = closed_forms(p)
    sol = fsolve(lambda z: best_response(p, *z) - z, x0=np.array([kc * 1.3, gc * 0.7, Yc * 1.2]), xtol=1e-14)
    errs.append(np.max(np.abs(sol - np.array([kc, gc, Yc])) / np.abs(np.array([kc, gc, Yc]))))
out += ["## (B) independent equilibrium (Kleinman policy iteration + fixed point, 60 random parameter sets)",
        f"- max relative error of (kappa, g, Ybar) against the closed forms: {max(errs):.2e}", ""]


# ======================================================================== (C) numerical identities
def system(p, s0_, sa_):
    kc, gc, Yc, kind_ = closed_forms(p)
    eta_, rho_, chi_, b_, ka_, ab_ = p
    M = np.array([[-kc, gc], [0.0, -ka_]])
    Sg = solve_continuous_lyapunov(M, -np.diag([s0_ ** 2, sa_ ** 2]))
    return kc, gc, M, Sg


maxdev = dict(cov=0.0, lev=0.0, cond=0, slope0=0.0, slope_lev=0.0, decomp=0.0, peak=0.0, corr0=0.0, passthrough=0.0)
for i in range(300):
    p = (10 ** rng.uniform(-0.7, 1.0), 10 ** rng.uniform(-2, -0.5), 10 ** rng.uniform(-2, -0.3),
         10 ** rng.uniform(-2, -0.3), 10 ** rng.uniform(-1, 1.3), 1.0)
    s0_, sa_ = 10 ** rng.uniform(-1.5, 0), 10 ** rng.uniform(-1.5, 0.5)
    kc, gc, M, Sg = system(p, s0_, sa_)
    b_, ka_ = p[3], p[4]
    Va, Cya = sa_ ** 2 / (2 * ka_), gc * sa_ ** 2 / (2 * ka_) / (kc + ka_)
    Vy = s0_ ** 2 / (2 * kc) + gc * Cya / kc
    maxdev["cov"] = max(maxdev["cov"], np.max(np.abs(np.array([Vy, Cya, Va]) - np.array([Sg[0, 0], Sg[0, 1], Sg[1, 1]]))) / Sg.max())
    cov_lev = Cya * (1 - b_ * gc / kc) - b_ * s0_ ** 2 / (2 * kc)
    l = np.array([-b_, 1.0]); e1 = np.array([1.0, 0.0])
    maxdev["lev"] = max(maxdev["lev"], abs(cov_lev - l @ Sg @ e1) / abs(l @ Sg @ l))
    cond = sa_ ** 2 * gc * (kc - b_ * gc) > b_ * s0_ ** 2 * ka_ * (kc + ka_)
    maxdev["cond"] += int(cond != (l @ Sg @ e1 > 0))
    Dm = np.diag([s0_ ** 2, sa_ ** 2])
    h = 1e-9 * min(1.0, s0_ ** 2 / (np.linalg.norm(M) ** 2 * np.linalg.norm(Sg) + 1e-300))
    blk = expm(np.block([[-M, Dm], [np.zeros((2, 2)), M.T]]) * h)
    Sig_h = blk[2:, 2:].T @ blk[:2, 2:]                       # Van Loan: int_0^h e^{Ms} D e^{M's} ds
    Eh = expm(M * h) - np.eye(2)
    Sh = Eh @ Sg @ Eh.T + Sig_h                               # exact Cov of h-increments
    maxdev["slope0"] = max(maxdev["slope0"], abs((l @ Sh @ e1) / Sh[0, 0] + b_) / b_)
    maxdev["corr0"] = max(maxdev["corr0"], abs((l @ Sh @ e1) / np.sqrt((l @ Sh @ l) * Sh[0, 0])
                                               + b_ * s0_ / np.sqrt(sa_ ** 2 + b_ ** 2 * s0_ ** 2)))
    maxdev["slope_lev"] = max(maxdev["slope_lev"], abs((l @ Sg @ e1) / Sg[0, 0] - (-b_ + Cya / Vy)))
    Sd = solve_continuous_lyapunov(M, -np.diag([0.0, sa_ ** 2]))
    maxdev["decomp"] = max(maxdev["decomp"], abs(l @ Sg @ l - (b_ ** 2 * s0_ ** 2 / (2 * kc) + l @ Sd @ l)) / (l @ Sg @ l))
    if abs(kc - ka_) > 1e-3:
        tau = np.linspace(0, 20 / min(kc, ka_), 400001)
        yresp = gc * (np.exp(-ka_ * tau) - np.exp(-kc * tau)) / (kc - ka_)
        maxdev["peak"] = max(maxdev["peak"], abs(tau[np.argmax(yresp)] - np.log(kc / ka_) / (kc - ka_)) * min(kc, ka_))
        lam_resp = np.exp(-ka_ * tau) - b_ * yresp
        integ = np.trapezoid(lam_resp, tau) / np.trapezoid(np.exp(-ka_ * tau), tau)
        maxdev["passthrough"] = max(maxdev["passthrough"], abs(integ - (1 - b_ * gc / kc)))
out += ["## (C) identities on 300 random parameter sets (maximum deviations)",
        f"- stationary covariances V_y, C_y_alpha, V_alpha vs Lyapunov: {maxdev['cov']:.1e}",
        f"- level covariance formula: {maxdev['lev']:.1e}; positivity condition misclassified: {maxdev['cond']} of 300",
        f"- short-horizon slope -> -b (exact increments, h -> 0): {maxdev['slope0']:.1e}; short-horizon correlation limit: {maxdev['corr0']:.1e}",
        f"- level slope = -b + C_y_alpha/V_y: {maxdev['slope_lev']:.1e}; variance decomposition: {maxdev['decomp']:.1e}",
        f"- peak time of the supply response (in units of 1/min speed): {maxdev['peak']:.1e}; integrated pass-through 1 - bg/kappa: {maxdev['passthrough']:.1e}",
        ""]

# ======================================================================== (D) temporal aggregation
par = pd.read_csv(RES / "13_two_factor_params.csv").set_index("parameter")["estimate"]
kap, kap_a, s_sh, r = (par["kappa (1/yr)"], par["kappa_a (1/yr)"], par["s = b/(chi+b)"], par["r = sigma_a/(b sigma0)"])
beta_fit = s_sh * kap * (kap + 0.05) / (0.05 + kap + kap_a)
M = np.array([[-kap, beta_fit], [0.0, -kap_a]])
Sg = solve_continuous_lyapunov(M, -np.diag([1.0, r * r]))
dday = 1 / 252
Ad = expm(M * dday)
Cd = Sg - Ad @ Sg @ Ad.T
Lc = np.linalg.cholesky(Cd)
ndays = 5 * 200000
z = np.zeros((ndays, 2))
eps = rng.standard_normal((ndays, 2)) @ Lc.T
z[0] = np.linalg.cholesky(Sg) @ rng.standard_normal(2)
for t in range(1, ndays):
    z[t] = Ad @ z[t - 1] + eps[t]
lam = z[:, 1] - z[:, 0]
yy = z[:, 0]


def weekly_moments(lw, yw):
    phi = lambda x: F.phi_hat(x[:-1], x[1:])
    Wm = np.column_stack([lw, yw])
    X0 = np.column_stack([np.ones(len(Wm) - 1), Wm[:-1]])
    coef, *_ = np.linalg.lstsq(X0, Wm[1:], rcond=None)
    Phi = coef[1:].T
    return dict(k_lam=-52 * np.log(phi(lw)), k_Y=-52 * np.log(phi(yw)), corr_lev=np.corrcoef(lw, yw)[0, 1],
                corr_chg=np.corrcoef(np.diff(lw), np.diff(yw))[0, 1], Phi_lam_Y=Phi[0, 1], Phi_Y_lam=Phi[1, 0],
                Phi_lam_lam=Phi[0, 0], Phi_Y_Y=Phi[1, 1])


point = weekly_moments(lam[::5], yy[::5])
avg = weekly_moments(lam.reshape(-1, 5).mean(axis=1), yy.reshape(-1, 5).mean(axis=1))
agg = pd.DataFrame([point, avg], index=["point-sampled weekly", "weekly means"]).T
out += ["## (D) fitted model simulated daily (1,000,000 days): point sampling vs weekly means", "", agg.round(4).to_string(), ""]

# ======================================================================== (E) empirical recomputations
india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
wk = d[["lam_back"]].join(poi[["log_short_detr", "seller_net_share", "fii_net_share"]], how="inner").resample("W-FRI").mean().dropna()
W = wk[["lam_back", "log_short_detr"]].to_numpy()
X0 = np.column_stack([np.ones(len(W) - 1), W[:-1]])
coef, *_ = np.linalg.lstsq(X0, W[1:], rcond=None)
Phi_ols = coef[1:].T
Wc = W - W.mean(axis=0)
G0 = Wc[:-1].T @ Wc[:-1] / (len(W) - 1)
G1 = Wc[1:].T @ Wc[:-1] / (len(W) - 1)
Phi_yw = G1 @ np.linalg.inv(G0)
dl, dy = np.diff(W[:, 0]), np.diff(W[:, 1])
Xs = np.column_stack([np.ones_like(dy), dy])
bs_, *_ = np.linalg.lstsq(Xs, dl, rcond=None)
e = dl - Xs @ bs_
XtXi = np.linalg.inv(Xs.T @ Xs)
Xe = Xs * e[:, None]
Sm = Xe.T @ Xe
for j in range(1, 9):
    Gm = Xe[j:].T @ Xe[:-j]
    Sm += (1 - j / 9) * (Gm + Gm.T)
se_nw = np.sqrt(np.diag(XtXi @ Sm @ XtXi))[1]
rows = []
rng2 = np.random.default_rng(16)
for col in ("log_short_detr", "seller_net_share", "fii_net_share"):
    Wv = wk[["lam_back", col]].to_numpy()
    n = len(Wv)

    def cf(A0, A1):
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(A0)), A0]), A1, rcond=None)
        return c[1:].T
    Ph = cf(Wv[:-1], Wv[1:])
    reps = np.array([cf(Wv[:-1][ib], Wv[1:][ib]) for ib in (F.block_indices(n - 1, 8, rng2) for _ in range(2000))])
    se = reps.std(axis=0, ddof=1)
    rows.append(dict(supply_measure=col, supply_to_premium=Ph[0, 1], t_s2p=Ph[0, 1] / se[0, 1],
                     premium_to_supply=Ph[1, 0], t_p2s=Ph[1, 0] / se[1, 0]))
Ev = pd.DataFrame(rows)
out += ["## (E) empirical recomputations",
        f"- weekly sample: {len(wk)} weeks, {wk.index[0].date()} to {wk.index[-1].date()}",
        f"- VAR(1) by OLS vs Yule-Walker, max abs difference of coefficients: {np.max(np.abs(Phi_ols - Phi_yw)):.2e} "
        f"(relative to max |Phi| {np.max(np.abs(Phi_ols)):.3f})",
        f"- 1-week change slope {bs_[1]:.5f}: Newey-West s.e. (8 lags) {se_nw:.5f}, t = {bs_[1] / se_nw:.2f}", "",
        "VAR cross effects with alternative supply measures (premium = trailing proxy):", Ev.round(4).to_string(index=False), ""]
Ev.to_csv(RES / "15_var_supply_measures.csv", index=False)
agg.to_csv(RES / "15_aggregation.csv")
(RES / "15_verify_two_factor.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
