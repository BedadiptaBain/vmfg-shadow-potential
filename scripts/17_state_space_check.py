#!/usr/bin/env python3
"""17 - Independent re-implementation of the latent-state estimation (cross-check of script 16).

Differences from script 16 by design: observables in raw units (demeaned, not standardised); the premium
loading fixed by the structural restriction lam - lbar = alpha - ytil (scaled state in premium units);
chi > 0 imposed through s = 1/(1 + exp(-theta)) in (0, 1). Same weekly end-of-week sampling.
Models: M1 (open interest loads on supply only), M2 (loads on both), M2b (beta = 0), M2u (beta free),
one-speed (kappa = kappa_a). Profile in s. Out-of-sample one-step forecasts (train to 2023-06-30)
against a VAR(1) and no-change. Implied lead-lag of observables. Usage: 17_state_space_check.py [END]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy.stats import chi2, kurtosis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402
from vmfg.empirical import realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
RHO, DT = 0.05, 1.0 / 52.0
END = sys.argv[1] if len(sys.argv) > 1 else "2026-08-14"
TAG = "" if END == "2026-08-14" else "_prebreak"
TRAIN_END = "2023-06-30"


def build(th, mode):
    k, ka = np.exp(th[0]), np.exp(th[1])
    if mode == "eq":
        beta = (1 / (1 + np.exp(-th[2]))) * k * (k + RHO) / (RHO + k + ka)
    elif mode == "free":
        beta = th[2]
    else:
        beta = 0.0
    M = np.array([[-k, beta], [0.0, -ka]])
    Sig = solve_continuous_lyapunov(M, -np.diag([np.exp(2 * th[3]), np.exp(2 * th[4])]))
    A = TF.expm_ut(M, DT)
    Q = Sig - A @ Sig @ A.T
    H = np.array([[-1.0, 1.0], [th[5], th[6]]])
    R = np.diag([np.exp(2 * th[7]), np.exp(2 * th[8])])
    return A, Q, H, R, Sig, beta


def negll(th, Z, mode, start=0, keep=False):
    try:
        A, Q, H, R, Sig, _ = build(th, mode)
        if np.min(np.linalg.eigvalsh(Q)) < -1e-14:
            return 1e10
    except Exception:
        return 1e10
    x, P = np.zeros(2), Sig.copy()
    ll, preds, xs, inn = 0.0, [], [], []
    for t, z in enumerate(Z):
        if t > 0:
            x, P = A @ x, A @ P @ A.T + Q
        preds.append(H @ x)
        ok = np.isfinite(z)
        if ok.any():
            Hk, Rk = H[ok], R[np.ix_(ok, ok)]
            v = z[ok] - Hk @ x
            Fk = Hk @ P @ Hk.T + Rk
            sgn, logdet = np.linalg.slogdet(Fk)
            if sgn <= 0:
                return 1e10
            Fi = np.linalg.inv(Fk)
            if t >= start:
                ll -= 0.5 * (logdet + v @ Fi @ v + ok.sum() * np.log(2 * np.pi))
            K = P @ Hk.T @ Fi
            x, P = x + K @ v, P - K @ Hk @ P
            e = np.full(2, np.nan); e[ok] = v / np.sqrt(np.diag(Fk)); inn.append(e)
        else:
            inn.append(np.full(2, np.nan))
        xs.append(x.copy())
    if keep:
        return -ll, np.array(preds), np.array(xs), np.array(inn)
    return -ll if np.isfinite(ll) else 1e10


def fit(Z, mode="eq", fix=None, starts=(), start=0):
    fix = fix or {}
    free = [i for i in range(9) if i not in fix]
    def full(u):
        th = np.zeros(9); th[free] = u
        for i, v in fix.items():
            th[i] = v
        return th
    best = None
    for st in starts:
        r = minimize(lambda u: negll(full(u), Z, mode, start), np.asarray(st)[free], method="Nelder-Mead",
                     options=dict(maxiter=6000, maxfev=6000, xatol=1e-6, fatol=1e-7))
        r = minimize(lambda u: negll(full(u), Z, mode, start), r.x, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return full(best.x), -best.fun


india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
d["pi_hat"] = d["IV2"] - RD.har_exante(d, "logs_median", return_forecast=True)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
wk = d[["pi_hat"]].join(poi[["log_short_detr"]], how="inner").resample("W-FRI").last()
wk = wk.loc[wk.dropna().index[0]:END]
Z = (wk - wk.mean()).to_numpy()
n = len(Z)
sP, sO = np.nanstd(Z[:, 0]), np.nanstd(Z[:, 1])
starts = []
for k0, ka0 in ((5.0, 15.0), (12.0, 3.5), (8.0, 8.0), (3.0, 30.0), (20.0, 5.0)):
    for lY, la in ((20.0, 0.0), (-20.0, 0.0), (0.0, 20.0)):
        starts.append([np.log(k0), np.log(ka0), 0.0, np.log(sP * np.sqrt(2 * k0) * 0.5), np.log(sP * np.sqrt(2 * ka0) * 0.7),
                       lY, la, np.log(sP * 0.5), np.log(sO * 0.8)])
th2, ll2 = fit(Z, "eq", starts=starts)
th1, ll1 = fit(Z, "eq", fix={6: 0.0}, starts=starts)
thb, llb = fit(Z, "zero", fix={2: 0.0}, starts=starts)
thu, llu = fit(Z, "free", starts=[np.r_[th2[:2], build(th2, "eq")[5], th2[3:]]] + starts[:5])
th_e, ll_e = None, -np.inf
for st in starts[:9]:
    r = minimize(lambda u: negll(np.r_[u[0], u[0], u[1:]], Z, "eq"), np.r_[st[0], st[2:]], method="Nelder-Mead",
                 options=dict(maxiter=6000, maxfev=6000))
    if -r.fun > ll_e:
        ll_e, th_e = -r.fun, np.r_[r.x[0], r.x[0], r.x[1:]]
lr = lambda a, b: 2 * (a - b)
tests = pd.DataFrame([
    dict(test="open interest loads on demand (M1 vs M2)", LR=lr(ll2, ll1), p=chi2.sf(lr(ll2, ll1), 1)),
    dict(test="demand moves supply (beta = 0 vs M2)", LR=lr(ll2, llb), p=chi2.sf(max(lr(ll2, llb), 0), 1)),
    dict(test="equilibrium restriction on beta (M2 vs beta free)", LR=lr(llu, ll2), p=chi2.sf(max(lr(llu, ll2), 0), 1)),
    dict(test="two speeds (kappa = kappa_a vs M2)", LR=lr(ll2, ll_e), p=chi2.sf(max(lr(ll2, ll_e), 0), 1)),
])
prof = []
for s in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99):
    _, lp = fit(Z, "eq", fix={2: np.log(s / (1 - s))}, starts=[th2])
    prof.append(dict(s=s, chi_over_b=(1 - s) / s, LR_vs_max=lr(ll2, lp)))
prof = pd.DataFrame(prof)
# out of sample
ntr = int(np.searchsorted(wk.index, pd.Timestamp(TRAIN_END), side="right"))
oos_rows = []
for name, mode, fix in (("M2", "eq", None), ("M1", "eq", {6: 0.0})):
    th_tr, _ = fit(Z[:ntr], mode, fix=fix, starts=starts[:9])
    _, preds, _, _ = negll(th_tr, Z, mode, keep=True)
    err = Z[ntr:] - preds[ntr:]
    oos_rows.append(dict(model=name, rmse_P=np.sqrt(np.nanmean(err[:, 0] ** 2)) / sP, rmse_O=np.sqrt(np.nanmean(err[:, 1] ** 2)) / sO,
                         pred_loglik=-negll(th_tr, Z, mode, start=ntr)))
okv = np.all(np.isfinite(np.c_[Z[:ntr - 1], Z[1:ntr]]), axis=1)
Bv, *_ = np.linalg.lstsq(np.c_[np.ones(okv.sum()), Z[:ntr - 1][okv]], Z[1:ntr][okv], rcond=None)
pv = np.c_[np.ones(n - ntr), Z[ntr - 1:n - 1]] @ Bv
ev = Z[ntr:] - pv
oos_rows.append(dict(model="VAR(1)", rmse_P=np.sqrt(np.nanmean(ev[:, 0] ** 2)) / sP, rmse_O=np.sqrt(np.nanmean(ev[:, 1] ** 2)) / sO, pred_loglik=np.nan))
en = Z[ntr:] - Z[ntr - 1:n - 1]
oos_rows.append(dict(model="no change", rmse_P=np.sqrt(np.nanmean(en[:, 0] ** 2)) / sP, rmse_O=np.sqrt(np.nanmean(en[:, 1] ** 2)) / sO, pred_loglik=np.nan))
oos = pd.DataFrame(oos_rows)
# lead-lag
A, Q, H, R, Sig, beta2 = build(th2, "eq")
Phi_m = (H @ A @ Sig @ H.T) @ np.linalg.inv(H @ Sig @ H.T + R)
Wv = wk.dropna().to_numpy()
cf, *_ = np.linalg.lstsq(np.c_[np.ones(len(Wv) - 1), Wv[:-1]], Wv[1:], rcond=None)
Phi_d = cf[1:].T
# standardised cross effects (scale-free comparison with script 16)
sd = np.nanstd(Wv, axis=0)
std = lambda Ph: Ph * sd[None, :] / sd[:, None]
leadlag = pd.DataFrame(dict(effect=["O -> future P", "P -> future O", "P persistence", "O persistence"],
                            empirical=[std(Phi_d)[0, 1], std(Phi_d)[1, 0], Phi_d[0, 0], Phi_d[1, 1]],
                            model_M2=[std(Phi_m)[0, 1], std(Phi_m)[1, 0], Phi_m[0, 0], Phi_m[1, 1]]))
_, _, xs, inn = negll(th2, Z, "eq", keep=True)
pr = dict(kappa=np.exp(th2[0]), kappa_a=np.exp(th2[1]), s=1 / (1 + np.exp(-th2[2])), beta=beta2, L_Y=th2[5], L_a=th2[6],
          sd_P_noise=np.exp(th2[7]), sd_O_noise=np.exp(th2[8]))
Vlat = float(np.array([-1, 1]) @ Sig @ np.array([-1, 1]))
share_noise_P = np.exp(2 * th2[7]) / (Vlat + np.exp(2 * th2[7]))
out = [f"# 17 - independent state-space check ({n} weeks to {wk.index[-1].date()})", "",
       f"log-likelihoods: M2 {ll2:.3f}, M1 {ll1:.3f}, beta=0 {llb:.3f}, beta free {llu:.3f}, one speed {ll_e:.3f}",
       "M2 parameters: " + ", ".join(f"{k}={v:.4g}" for k, v in pr.items()),
       f"noise share of the premium observable: {share_noise_P:.3f}", "",
       tests.round(4).to_string(index=False), "", "profile in s:", prof.round(3).to_string(index=False), "",
       f"out of sample (train to {TRAIN_END}, {n - ntr} weeks; RMSE in s.d. units):", oos.round(4).to_string(index=False), "",
       "lead-lag (cross effects standardised):", leadlag.round(4).to_string(index=False), "",
       f"innovation excess kurtosis: premium {kurtosis(inn[np.isfinite(inn[:,0]),0]):.1f}, open interest {kurtosis(inn[np.isfinite(inn[:,1]),1]):.1f}"]
for nm, df in (("tests", tests), ("profile", prof), ("oos", oos), ("leadlag", leadlag)):
    df.to_csv(RES / f"17_{nm}{TAG}.csv", index=False)
pd.Series(pr).to_csv(RES / f"17_params{TAG}.csv")
(RES / f"17_state_space_check{TAG}.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
