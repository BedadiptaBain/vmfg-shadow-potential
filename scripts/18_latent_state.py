#!/usr/bin/env python3
"""18 - Latent-state test of the two-factor clearing mechanism (exact weekly-mean aggregation).

State (latent, scaled): xi = (ytil, alpha), supply and demand, with unit noises (normalisation):
    d ytil  = (-kappa ytil + beta alpha) dt + dW^0,    d alpha = -kappa_a alpha dt + dW^a.
Observations are weekly MEANS of daily values (5 days per week), handled exactly by the augmented
state x_k = (xi at the end of week k, mean of xi over week k). Observables (standardised):
    P  ex-ante premium estimate            = pi_y ytil + pi_a alpha + e_P
    S  net short share of the writers      = l_S  ytil               + e_S     (l_S > 0: sign of supply)
    O  log total short OI, detrended       = o_y  ytil + o_a  alpha + e_O
Normalisation pi_a > 0 fixes the sign of demand. With free scales, lam = a - bY has no equality content;
its testable content is in the signs:  b > 0 <=> pi_y < 0;  g > 0 <=> beta > 0;  chi > 0 <=> s in (0,1)
with s = beta (-pi_y)(rho+kappa+kappa_a)/(pi_a kappa (kappa+rho)); demand footprint in OI <=> o_a > 0.
Also: nested curvature (extended Kalman filter, gamma ytil^2 in P), one-factor comparison, out-of-sample
log scores, and the same estimation on the sample before the November 2024 rule changes.
Sample end for the pre-break run: env VMFG_SS_END (default: full sample).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.optimize import minimize
from scipy.stats import chi2, kurtosis, norm, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
RHO, DELTA, NDAY = 0.05, 1.0 / 260.0, 5
END = os.environ.get("VMFG_SS_END")
ARP = os.environ.get("VMFG_SS_ARP") == "1"
TAG = ("_prebreak" if END else "") + ("_arP" if ARP else "")
rng = np.random.default_rng(18)

india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
d["pi_hat"] = d["IV2"] - RD.har_exante(d, "logs_median", return_forecast=True)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
wk = d[["pi_hat"]].join(poi[["seller_net_share", "log_short_detr"]], how="inner").resample("W-FRI").mean().dropna()
if END:
    wk = wk.loc[:END]
Yraw = wk.to_numpy()
Y = (Yraw - Yraw.mean(axis=0)) / Yraw.std(axis=0)
dates, T = wk.index, len(wk)


def aggregated(M):
    n = M.shape[0]
    A = expm(M * DELTA)
    Sig = solve_continuous_lyapunov(M, -np.eye(n))
    Qd = Sig - A @ Sig @ A.T
    Ap = [np.linalg.matrix_power(A, j) for j in range(NDAY + 1)]
    Abar = sum(Ap[1:]) / NDAY
    Bi = [sum(Ap[j - i] for j in range(i, NDAY + 1)) / NDAY for i in range(1, NDAY + 1)]
    Cee = sum(Ap[NDAY - i] @ Qd @ Ap[NDAY - i].T for i in range(1, NDAY + 1))
    Cmm = sum(B @ Qd @ B.T for B in Bi)
    Cem = sum(Ap[NDAY - i] @ Qd @ Bi[i - 1].T for i in range(1, NDAY + 1))
    F = np.zeros((2 * n, 2 * n)); F[:n, :n] = Ap[NDAY]; F[n:, :n] = Abar
    W = np.block([[Cee, Cem], [Cem.T, Cmm]])
    return F, W, solve_discrete_lyapunov(F, W)


def augment_arP(F, W, P0, phi, sig):
    """Append an AR(1) measurement-error state for the premium: e' = phi e + eta, Var(eta) = sig^2."""
    k = F.shape[0]
    Fa = np.zeros((k + 1, k + 1)); Fa[:k, :k] = F; Fa[k, k] = phi
    Wa = np.zeros((k + 1, k + 1)); Wa[:k, :k] = W; Wa[k, k] = sig ** 2
    Pa = np.zeros((k + 1, k + 1)); Pa[:k, :k] = P0; Pa[k, k] = sig ** 2 / (1 - phi ** 2)
    return Fa, Wa, Pa


def kalman(y, F, W, P0, Hm, Rd, gamma=0.0, full=False, arP=None):
    if arP is not None:
        F, W, P0 = augment_arP(F, W, P0, *arP)
    n2 = F.shape[0] - (1 if arP is not None else 0); n = n2 // 2
    H = np.zeros((Hm.shape[0], F.shape[0])); H[:, n:n2] = Hm
    if arP is not None:
        H[0, n2] = 1.0
    n2 = F.shape[0]
    x, P, R = np.zeros(n2), P0.copy(), np.diag(Rd)
    ll = np.zeros(len(y)); inn = np.zeros_like(y); st = np.zeros((len(y), min(n, 2)))
    for t in range(len(y)):
        x = F @ x; P = F @ P @ F.T + W
        Ht, yh = H.copy(), H @ x
        if gamma:
            yh = yh.copy(); yh[0] += gamma * (x[n] ** 2 + P[n, n]); Ht[0, n] += 2 * gamma * x[n]
        v = y[t] - yh
        S = Ht @ P @ Ht.T + R
        K = np.linalg.solve(S, Ht @ P).T
        x = x + K @ v; P = P - K @ Ht @ P
        ll[t] = -0.5 * (np.linalg.slogdet(S)[1] + v @ np.linalg.solve(S, v) + len(v) * np.log(2 * np.pi))
        inn[t] = v / np.sqrt(np.diag(S)); st[t] = x[n:n + 2] if len(x) > n + 1 else x[n:]
    return (ll, inn, st) if full else ll


NB = 13 if ARP else 11


def q2(th):
    return dict(kappa=np.exp(th[0]), kappa_a=np.exp(th[1]), beta=th[2], pi_y=th[3], pi_a=np.exp(th[4]),
                l_S=np.exp(th[5]), o_y=th[6], o_a=th[7], R=np.exp(2 * np.asarray(th[8:11])),
                arP=(float(np.tanh(th[11])), float(np.exp(th[12]))) if ARP else None,
                gamma=th[NB] if len(th) > NB else 0.0)


def sys2(q):
    F, W, P0 = aggregated(np.array([[-q["kappa"], q["beta"]], [0.0, -q["kappa_a"]]]))
    return F, W, P0, np.array([[q["pi_y"], q["pi_a"]], [q["l_S"], 0.0], [q["o_y"], q["o_a"]]])


def nll2(th, y):
    q = q2(th)
    if not (0.05 < q["kappa"] < 300 and 0.05 < q["kappa_a"] < 300) or abs(q["kappa"] - q["kappa_a"]) < 1e-6:
        return 1e10
    try:
        F, W, P0, Hm = sys2(q)
        return -kalman(y, F, W, P0, Hm, q["R"], q["gamma"], arP=q["arP"]).sum()
    except (np.linalg.LinAlgError, ValueError):
        return 1e10


def q1(th):
    return dict(kappa=np.exp(th[0]), pi_y=th[1], l_S=np.exp(th[2]), o_y=th[3], R=np.exp(2 * np.asarray(th[4:7])))


def nll1(th, y):
    q = q1(th)
    if not 0.05 < q["kappa"] < 300:
        return 1e10
    F, W, P0 = aggregated(np.array([[-q["kappa"]]]))
    return -kalman(y, F, W, P0, np.array([[q["pi_y"]], [q["l_S"]], [q["o_y"]]]), q["R"]).sum()


def fit(nll, starts, y):
    best = None
    for x0 in starts:
        r = minimize(nll, x0, args=(y,), method="L-BFGS-B", options=dict(maxiter=500))
        if best is None or r.fun < best.fun:
            best = r
    return minimize(nll, best.x, args=(y,), method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-6, fatol=1e-8))


def hessian(f, x, h=1e-4):
    k = len(x); Hs = np.zeros((k, k)); I = np.eye(k) * h
    for i in range(k):
        for j in range(i, k):
            Hs[i, j] = Hs[j, i] = (f(x + I[i] + I[j]) - f(x + I[i] - I[j]) - f(x - I[i] + I[j]) + f(x - I[i] - I[j])) / (4 * h * h)
    return Hs


base2 = np.array([np.log(4.0), np.log(15.0), 0.5, -0.3, np.log(0.5), np.log(0.5), 0.1, 0.2, np.log(0.6), np.log(0.6), np.log(0.6)]
                 + ([np.arctanh(0.7), np.log(0.4)] if ARP else []))
S2 = [base2] + [base2 + rng.normal(0, 0.7, len(base2)) for _ in range(5)] + \
     [np.r_[np.log(15.0), np.log(3.0), base2[2:]]]            # also try demand slower than supply
base1 = np.array([np.log(4.0), -0.3, np.log(0.5), 0.2, np.log(0.6), np.log(0.6), np.log(0.6)])
S1 = [base1] + [base1 + rng.normal(0, 0.7, 7) for _ in range(4)]

r2 = fit(nll2, S2, Y)
th = r2.x
cov = np.linalg.pinv(hessian(lambda z: nll2(z, Y), th))
draws = rng.multivariate_normal(th, cov, size=4000)


def derived(t):
    q = q2(t)
    s_imp = q["beta"] * (-q["pi_y"]) * (RHO + q["kappa"] + q["kappa_a"]) / (q["pi_a"] * q["kappa"] * (q["kappa"] + RHO))
    return dict(kappa=q["kappa"], kappa_a=q["kappa_a"], beta=q["beta"], pi_y=q["pi_y"], pi_a=q["pi_a"], l_S=q["l_S"],
                o_y=q["o_y"], o_a=q["o_a"], s_implied=s_imp, kappa_a_minus_kappa=q["kappa_a"] - q["kappa"])


est = derived(th)
Dd = pd.DataFrame([derived(t) for t in draws])
par = pd.DataFrame(dict(estimate=pd.Series(est), se=Dd.std(), q025=Dd.quantile(0.025), q975=Dd.quantile(0.975)))
pz = lambda k: est[k] / par.loc[k, "se"]
tests = pd.DataFrame([
    dict(prediction="b > 0 (premium falls with supply: pi_y < 0)", estimate=est["pi_y"], se=par.loc["pi_y", "se"], p_value=float(norm.cdf(-(-pz("pi_y"))) if False else norm.sf(-pz("pi_y")))),
    dict(prediction="g > 0 (demand raises supply: beta > 0)", estimate=est["beta"], se=par.loc["beta", "se"], p_value=float(norm.sf(pz("beta")))),
    dict(prediction="demand footprint in total OI (o_a > 0)", estimate=est["o_a"], se=par.loc["o_a", "se"], p_value=float(norm.sf(pz("o_a")))),
    dict(prediction="chi > 0 (s_implied in (0,1)): share of draws outside", estimate=est["s_implied"], se=par.loc["s_implied", "se"],
         p_value=float(np.mean((Dd["s_implied"] <= 0) | (Dd["s_implied"] >= 1)))),
    dict(prediction="two speeds (kappa_a > kappa): share of draws with kappa_a <= kappa", estimate=est["kappa_a_minus_kappa"],
         se=par.loc["kappa_a_minus_kappa", "se"], p_value=float(np.mean(Dd["kappa_a_minus_kappa"] <= 0))),
])
# one-sided p-value for pi_y < 0: probability of pi_y >= 0 under the asymptotic normal
tests.loc[0, "p_value"] = float(norm.sf(-pz("pi_y")))

F, W, P0, Hm = sys2(q2(th))
qq = q2(th)
if ARP:
    Fx, Wx, P0x = augment_arP(F, W, P0, *qq["arP"])
    H = np.zeros((3, 5)); H[:, 2:4] = Hm; H[0, 4] = 1.0
else:
    Fx, Wx, P0x = F, W, P0
    H = np.zeros((3, 4)); H[:, 2:] = Hm
G0 = H @ P0x @ H.T + np.diag(qq["R"]); G1 = H @ Fx @ P0x @ H.T
Phi_m = G1 @ np.linalg.inv(G0)
cf, *_ = np.linalg.lstsq(np.column_stack([np.ones(T - 1), Y[:-1]]), Y[1:], rcond=None)
Phi_d = cf[1:].T
reps = []
for _ in range(2000):
    s0 = rng.integers(0, T - 8, -(-(T - 1) // 8)); ix = (s0[:, None] + np.arange(8)).ravel()[:T - 1]
    c2, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(ix)), Y[:-1][ix]]), Y[1:][ix], rcond=None); reps.append(c2[1:].T)
reps = np.array(reps)
nm = ["P", "S", "O"]
VAR = pd.DataFrame([dict(effect=f"{nm[j]} -> future {nm[i]}", data=Phi_d[i, j], se=reps[:, i, j].std(ddof=1), model=Phi_m[i, j])
                    for i in range(3) for j in range(3)])
VAR["z"] = (VAR["data"] - VAR["model"]) / VAR["se"]

rc = min((minimize(lambda z: nll2(z, Y), np.append(th, g0), method="Nelder-Mead", options=dict(maxiter=8000, xatol=1e-6, fatol=1e-8))
          for g0 in (0.0, -0.1, 0.1)), key=lambda r: r.fun)
LR_c, gamma_hat = 2 * (r2.fun - rc.fun), rc.x[-1]
r1 = fit(nll1, S1, Y)
LR_21 = 2 * (r1.fun - r2.fun)

split = int(np.searchsorted(dates, pd.Timestamp("2024-06-28"))) if not END else int(0.67 * T)
r2t, r1t = fit(nll2, S2[:4], Y[:split]), fit(nll1, S1[:3], Y[:split])
qa = q2(r2t.x); Fa, Wa, P0a, Hma = sys2(qa); ll2 = kalman(Y, Fa, Wa, P0a, Hma, qa["R"], arP=qa["arP"])
qb = q1(r1t.x); Fb, Wb, P0b = aggregated(np.array([[-qb["kappa"]]]))
ll1 = kalman(Y, Fb, Wb, P0b, np.array([[qb["pi_y"]], [qb["l_S"]], [qb["o_y"]]]), qb["R"])
X = np.column_stack([np.ones(split - 1), Y[:split - 1]]); cv, *_ = np.linalg.lstsq(X, Y[1:split], rcond=None)
res = Y[1:split] - X @ cv; Sv = res.T @ res / (len(res) - 4)
llv = np.array([-0.5 * (np.linalg.slogdet(Sv)[1] + (Y[t] - np.r_[1, Y[t - 1]] @ cv) @ np.linalg.solve(Sv, Y[t] - np.r_[1, Y[t - 1]] @ cv)
                        + 3 * np.log(2 * np.pi)) for t in range(split, T)])


def dm(x, lag=4):
    e = x - x.mean(); s = e @ e / len(e)
    for j in range(1, lag + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / len(e)
    return float(x.mean() / np.sqrt(s / len(e)))


oos = dict(test_weeks=T - split, score_two=float(ll2[split:].sum()), score_one=float(ll1[split:].sum()), score_var=float(llv.sum()),
           dm_two_vs_one=dm(ll2[split:] - ll1[split:]), dm_two_vs_var=dm(ll2[split:] - llv))
_, inn, _ = kalman(Y, F, W, P0, Hm, q2(th)["R"], full=True, arP=q2(th)["arP"])
innov = pd.DataFrame(dict(series=nm, skew=[skew(inn[:, i]) for i in range(3)], excess_kurtosis=[kurtosis(inn[:, i]) for i in range(3)]))
summary = pd.DataFrame([dict(arP_phi=(q2(th)["arP"][0] if ARP else 0.0), weeks=T, start=str(dates[0].date()), end=str(dates[-1].date()), loglik_two=-r2.fun, loglik_one=-r1.fun,
                             LR_two_vs_one=LR_21, p_two_vs_one=float(chi2.sf(LR_21, 4)), gamma=gamma_hat, LR_curvature=LR_c,
                             p_curvature=float(chi2.sf(max(LR_c, 0), 1)), **oos)])
for nmf, df in (("params", par), ("tests", tests), ("var", VAR), ("summary", summary), ("innovations", innov)):
    df.to_csv(RES / f"18_{nmf}{TAG}.csv", index=(nmf == "params"))
pd.set_option("display.width", 220)
print(summary.T.to_string()); print(par.round(4).to_string()); print(tests.round(4).to_string(index=False))
print(VAR.round(3).to_string(index=False)); print(innov.round(3).to_string(index=False))
