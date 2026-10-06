#!/usr/bin/env python3
"""19 - Latent two-factor state-space estimation of the clearing model (daily NIFTY, 2020-2026).

Latent state z = (y, alpha): y = Y - Ybar in units of the supply measure, alpha = a - abar (demand).
    dy     = (-kappa y + g alpha) dt + sigma0 dW^0
    dalpha = -kappa_a alpha dt + sigma_a dW^a
Structural restriction (Theorem 7.3): b g = s kappa (kappa + rho) / (rho + kappa + kappa_a), s = b/(chi+b) in (0,1).
Observations (daily, trading-time step 1/252, so no averaging problem):
    P_t = mu_P + alpha_t - b y_t + e^P_t       (ex-ante premium estimate)
    S_t = mu_S + y_t + e^S_t                    (client + pro net short share, weekday-demeaned)
Parameterisation: kappa, kappa_a, b and the sigmas by logs (b > 0 imposed); s by logistic (chi > 0 imposed).
Estimation: exact Gaussian likelihood by the Kalman filter; QML sandwich standard errors.
Outputs: estimates; profile likelihood in s; smoothed shock tails with a two-component normal mixture;
the local-slope curvature test; out-of-sample predictive log scores (fit 2020-2023, evaluate 2024-2026).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402
from vmfg.empirical import inference as FI, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
RHO, DT = 0.05, 1.0 / 252.0
NAMES = ["kappa", "kappa_a", "b", "s", "sigma0", "sigma_a", "sigma_P", "sigma_S", "mu_P", "mu_S"]


PREM = sys.argv[1] if len(sys.argv) > 1 else "pi_hat"
SUPP = sys.argv[2] if len(sys.argv) > 2 else "seller_net_share"
TAG = "" if (PREM, SUPP) == ("pi_hat", "seller_net_share") else f"_{PREM}_{SUPP}"


def load_data():
    india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
    d = RD.add_constructions(india)
    d["pi_hat"] = d["IV2"] - RD.har_exante(d, "logs_median", return_forecast=True)
    poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
    df = d[[PREM]].join(poi[[SUPP]], how="inner").loc["2020-01-01":].rename(columns={PREM: "pi_hat", SUPP: "seller_net_share"})
    col = "seller_net_share"
    df[col] = df[col] - df.groupby(df.index.dayofweek)[col].transform("mean") + df[col].mean()
    return df


def unpack(th):
    return dict(kappa=math.exp(th[0]), kappa_a=math.exp(th[1]), b=math.exp(th[2]), s=1 / (1 + math.exp(-th[3])),
                sigma0=math.exp(th[4]), sigma_a=math.exp(th[5]), sigma_P=math.exp(th[6]), sigma_S=math.exp(th[7]),
                mu_P=th[8], mu_S=th[9])


def system(p):
    beta = p["s"] * p["kappa"] * (p["kappa"] + RHO) / (RHO + p["kappa"] + p["kappa_a"])
    g = beta / p["b"]
    M = np.array([[-p["kappa"], g], [0.0, -p["kappa_a"]]])
    Sig = solve_continuous_lyapunov(M, -np.diag([p["sigma0"] ** 2, p["sigma_a"] ** 2]))
    A = TF.expm_ut(M, DT)
    return M, Sig, A, Sig - A @ Sig @ A.T, g


def kalman(th, P_obs, S_obs, keep=False, one_factor=False):
    p = unpack(th)
    if one_factor:
        p["sigma_a"] = 1e-9
    M, Sig, A, Q, g = system(p)
    a11, a12, a22 = A[0, 0], A[0, 1], A[1, 1]
    q11, q12, q22 = Q[0, 0], Q[0, 1], Q[1, 1]
    m1 = m2 = 0.0
    p11, p12, p22 = Sig[0, 0], Sig[0, 1], Sig[1, 1]
    b, rP, rS, muP, muS = p["b"], p["sigma_P"] ** 2, p["sigma_S"] ** 2, p["mu_P"], p["mu_S"]
    n = len(P_obs)
    ll_t = np.zeros(n)
    predP = np.full((n, 2), np.nan)
    store = [] if keep else None
    for t in range(n):
        if t > 0:
            m1, m2 = a11 * m1 + a12 * m2, a22 * m2
            t11, t12, t22 = a11 * p11 + a12 * p12, a11 * p12 + a12 * p22, a22 * p22
            p11, p12, p22 = t11 * a11 + t12 * a12 + q11, t12 * a22 + q12, t22 * a22 + q22
        if keep:
            store.append((m1, m2, p11, p12, p22))
        ll = 0.0
        hm, h1, h2 = -b * m1 + m2, -b * p11 + p12, -b * p12 + p22
        Fp = -b * h1 + h2 + rP
        predP[t] = (muP + hm, Fp)
        yP = P_obs[t]
        if yP == yP:
            v = yP - muP - hm
            k1, k2 = h1 / Fp, h2 / Fp
            m1, m2 = m1 + k1 * v, m2 + k2 * v
            p11, p12, p22 = p11 - k1 * h1, p12 - k1 * h2, p22 - k2 * h2
            ll -= 0.5 * (math.log(2 * math.pi * Fp) + v * v / Fp)
        yS = S_obs[t]
        if yS == yS:
            v = yS - muS - m1
            Fs = p11 + rS
            k1, k2 = p11 / Fs, p12 / Fs
            m1, m2 = m1 + k1 * v, m2 + k2 * v
            p11, p12, p22 = p11 - k1 * p11, p12 - k1 * p12, p22 - k2 * p12
            ll -= 0.5 * (math.log(2 * math.pi * Fs) + v * v / Fs)
        ll_t[t] = ll
        if keep:
            store.append((m1, m2, p11, p12, p22))
    return ll_t.sum(), ll_t, predP, store, (A, Q, p)


def fit(P_obs, S_obs, starts, fixed=None, one_factor=False, maxiter=800):
    free = [i for i in range(10) if not (fixed and i in fixed)]

    def full(x):
        th = np.zeros(10)
        th[free] = x
        for i, v in (fixed or {}).items():
            th[i] = v
        return th

    def obj(x):
        try:
            v = -kalman(full(x), P_obs, S_obs, one_factor=one_factor)[0]
        except (ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
            return 1e12
        return v if np.isfinite(v) else 1e12

    best = None
    for st in starts:
        x0 = np.asarray(st, dtype=float)[free]
        r = minimize(obj, x0, method="L-BFGS-B", options=dict(maxiter=maxiter))
        if r.fun < 1e11 and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError("no start produced a finite likelihood")
    return full(best.x), -best.fun


def sandwich(th, P_obs, S_obs, h=1e-4):
    k = len(th)
    scores = np.zeros((len(P_obs), k))
    for i in range(k):
        e = np.zeros(k); e[i] = h
        scores[:, i] = (kalman(th + e, P_obs, S_obs)[1] - kalman(th - e, P_obs, S_obs)[1]) / (2 * h)
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            f = [kalman(th + si * ei + sj * ej, P_obs, S_obs)[0] for si, sj in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
            Hm[i, j] = Hm[j, i] = (f[0] - f[1] - f[2] + f[3]) / (4 * h * h)
    Hinv = np.linalg.pinv(-Hm)
    return Hinv @ (scores.T @ scores) @ Hinv


def smooth(th, P_obs, S_obs):
    _, _, _, store, (A, Q, p) = kalman(th, P_obs, S_obs, keep=True)
    n = len(P_obs)
    mp = np.array([store[2 * t][:2] for t in range(n)])
    Pp = np.array([[[store[2 * t][2], store[2 * t][3]], [store[2 * t][3], store[2 * t][4]]] for t in range(n)])
    mf = np.array([store[2 * t + 1][:2] for t in range(n)])
    Pf = np.array([[[store[2 * t + 1][2], store[2 * t + 1][3]], [store[2 * t + 1][3], store[2 * t + 1][4]]] for t in range(n)])
    ms = mf.copy()
    Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        J = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1])
        ms[t] = mf[t] + J @ (ms[t + 1] - mp[t + 1])
        Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
    eta = ms[1:] - ms[:-1] @ A.T
    Lc = np.linalg.cholesky(Q)
    return ms, np.linalg.solve(Lc, eta.T).T


def mixture2(x, iters=400):
    x = np.asarray(x)
    x = x / x.std()
    p, s1, s2 = 0.1, 0.7, 3.0
    for _ in range(iters):
        f1 = (1 - p) * stats.norm.pdf(x, 0, s1)
        f2 = p * stats.norm.pdf(x, 0, s2)
        w = f2 / (f1 + f2)
        p = w.mean()
        s1 = math.sqrt(np.sum((1 - w) * x ** 2) / np.sum(1 - w))
        s2 = math.sqrt(np.sum(w * x ** 2) / np.sum(w))
    llm = np.sum(np.log((1 - p) * stats.norm.pdf(x, 0, s1) + p * stats.norm.pdf(x, 0, s2)))
    lln = np.sum(stats.norm.logpdf(x, 0, x.std()))
    return p, s2 / s1, 2 * (llm - lln)


def nw_ols(y, X, lag):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lag + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lag + 1)) * (G + G.T)
    return beta, np.sqrt(np.diag(XtXi @ S @ XtXi))


def main():
    df = load_data()
    P_obs = df["pi_hat"].to_numpy(dtype=float)
    S_obs = df["seller_net_share"].to_numpy(dtype=float)
    out = ["# 19 - Latent two-factor state-space estimation (daily)", "",
           f"sample {df.index[0].date()} to {df.index[-1].date()}, {len(df)} sessions; P = {PREM}, S = {SUPP} (weekday-demeaned)", ""]
    sdP, sdS, mP, mS = np.nanstd(P_obs), np.nanstd(S_obs), np.nanmean(P_obs), np.nanmean(S_obs)
    starts = []
    for k, ka, s in ((6.0, 15.0, 0.5), (3.0, 40.0, 0.8), (10.0, 8.0, 0.3)):
        starts.append([math.log(k), math.log(ka), math.log(0.3 * sdP / sdS), math.log(s / (1 - s)),
                       math.log(sdS * math.sqrt(2 * k)), math.log(sdP * math.sqrt(2 * ka)),
                       math.log(0.3 * sdP), math.log(0.3 * sdS), mP, mS])
    th, ll = fit(P_obs, S_obs, starts)
    V = sandwich(th, P_obs, S_obs)
    se_th = np.sqrt(np.clip(np.diag(V), 0, None))
    p = unpack(th)
    jac = np.array([p["kappa"], p["kappa_a"], p["b"], p["s"] * (1 - p["s"]), p["sigma0"], p["sigma_a"],
                    p["sigma_P"], p["sigma_S"], 1.0, 1.0])
    est = pd.DataFrame(dict(parameter=NAMES, estimate=[p[k] for k in NAMES], se=jac * se_th))
    chib = (1 - p["s"]) / p["s"]
    se_chib = se_th[3] * chib
    M, Sig, A, Q, g = system(p)
    l = np.array([-p["b"], 1.0])
    V_lam = l @ Sig @ l
    share_dem = 1 - p["b"] ** 2 * p["sigma0"] ** 2 / (2 * p["kappa"]) / V_lam
    corr_lev = (l @ Sig @ np.array([1.0, 0.0])) / math.sqrt(V_lam * Sig[0, 0])
    out += ["## Estimates (QML sandwich s.e.)", est.round(5).to_string(index=False), "",
            f"- log-likelihood {ll:.2f}; chi/b = (1-s)/s = {chib:.4f} (s.e. {se_chib:.4f}); g = {g:.4f}",
            f"- demand share of premium variance {share_dem:.3f}; implied level correlation of premium and supply {corr_lev:.3f}", ""]

    prof = []
    for s_fix in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999):
        th_s, ll_s = fit(P_obs, S_obs, [th], fixed={3: math.log(s_fix / (1 - s_fix))}, maxiter=400)
        prof.append(dict(s=s_fix, chi_over_b=(1 - s_fix) / s_fix, loglik=ll_s, LR=2 * (ll - ll_s)))
    prof = pd.DataFrame(prof)
    out += ["## Profile likelihood in s (LR against the maximum; chi2(1) 5% critical value 3.84)",
            prof.round(3).to_string(index=False), ""]

    ms, std = smooth(th, P_obs, S_obs)
    rows = []
    for name, x in (("supply (W^0)", std[:, 0]), ("demand (W^a)", std[:, 1])):
        pj, ratio, lr = mixture2(x)
        rows.append(dict(shock=name, excess_kurtosis=stats.kurtosis(x), skew=stats.skew(x),
                         jb_p=stats.jarque_bera(x).pvalue, jump_prob=pj, jump_sd_ratio=ratio, LR_mixture=lr))
    tails = pd.DataFrame(rows)
    out += ["## Smoothed state shocks (standardised)", tails.round(4).to_string(index=False), ""]

    curv = []
    for freq, lag in (("daily", 10), ("weekly", 8)):
        z = df[["pi_hat", "seller_net_share"]]
        if freq == "weekly":
            z = z.resample("W-FRI").mean()
        z = z.dropna()
        dP, dS = np.diff(z["pi_hat"].to_numpy()), np.diff(z["seller_net_share"].to_numpy())
        lev = z["seller_net_share"].to_numpy()[:-1] - z["seller_net_share"].mean()
        beta, se = nw_ols(dP, np.column_stack([np.ones_like(dS), dS, dS * lev]), lag)
        b_hat, c_hat = -beta[1], -beta[2]
        curv.append(dict(frequency=freq, n=len(dP), b_hat=b_hat, se_b=se[1], t_b=b_hat / se[1], c_hat=c_hat,
                         se_c=se[2], t_c=c_hat / se[2],
                         min_local_slope=min(b_hat + c_hat * lev.min(), b_hat + c_hat * lev.max())))
    curv = pd.DataFrame(curv)
    out += ["## Curvature: premium changes on supply changes, local slope -(b + c (S - Sbar)), Newey-West s.e.",
            curv.round(5).to_string(index=False), ""]

    cut = int(np.searchsorted(df.index.values, np.datetime64("2024-01-01")))
    th_in, _ = fit(P_obs[:cut], S_obs[:cut], [th], maxiter=400)
    th_one, _ = fit(P_obs[:cut], S_obs[:cut], [th], one_factor=True, maxiter=400)
    pred2 = kalman(th_in, P_obs, S_obs)[2]
    pred1 = kalman(th_one, P_obs, S_obs, one_factor=True)[2]
    x = P_obs[:cut]
    ok = np.isfinite(x[:-1]) & np.isfinite(x[1:])
    Xa = np.column_stack([np.ones(ok.sum()), x[:-1][ok]])
    ca, *_ = np.linalg.lstsq(Xa, x[1:][ok], rcond=None)
    sa2 = np.var(x[1:][ok] - Xa @ ca)
    sc = []
    for t in range(cut, len(P_obs)):
        if np.isfinite(P_obs[t]) and np.isfinite(P_obs[t - 1]):
            sc.append((stats.norm.logpdf(P_obs[t], pred2[t, 0], math.sqrt(pred2[t, 1])),
                       stats.norm.logpdf(P_obs[t], pred1[t, 0], math.sqrt(pred1[t, 1])),
                       stats.norm.logpdf(P_obs[t], ca[0] + ca[1] * P_obs[t - 1], math.sqrt(sa2))))
    sc = np.array(sc)
    oos = []
    for j, name in ((1, "one-factor state space"), (2, "univariate AR(1)")):
        mu, se_ = FI.nw_mean(sc[:, 0] - sc[:, j], 10)
        oos.append(dict(benchmark=name, n=len(sc), mean_log_score_gain=mu, se=se_, t=mu / se_))
    oos = pd.DataFrame(oos)
    out += ["## Out-of-sample: fit 2020-2023, one-step predictive log score for the premium in 2024-2026 "
            "(gain of the two-factor model)", oos.round(5).to_string(index=False), ""]

    est.to_csv(RES / f"19_ss_estimates{TAG}.csv", index=False)
    prof.to_csv(RES / f"19_ss_profile_s{TAG}.csv", index=False)
    tails.to_csv(RES / f"19_ss_tails{TAG}.csv", index=False)
    curv.to_csv(RES / f"19_curvature{TAG}.csv", index=False)
    oos.to_csv(RES / f"19_oos{TAG}.csv", index=False)
    pd.DataFrame([dict(loglik=ll, chi_over_b=chib, se_chi_over_b=se_chib, g=g, demand_share=share_dem,
                       corr_levels=corr_lev, n=len(df), start=str(df.index[0].date()), end=str(df.index[-1].date()))]
                 ).to_csv(RES / f"19_ss_summary{TAG}.csv", index=False)
    (RES / f"19_state_space_daily{TAG}.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
