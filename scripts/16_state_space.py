#!/usr/bin/env python3
"""16 - Latent two-factor clearing model: state-space estimation and structural tests.

State (scaled, weekly exact discretisation of Theorem 7.3; unit supply noise):
    xi = (ytil, alpha),  d ytil = (-kappa ytil + beta alpha) dt + dW^0,  d alpha = -kappa_a alpha dt + r dW^a,
    beta = s kappa (kappa + rho) / (rho + kappa + kappa_a),  s = b/(chi + b) in (0, 1) (logistic: chi > 0).
Observations (end-of-week snapshots, standardised):
    P  ex-ante premium estimate   = c_P (alpha - ytil) + e_P        (b > 0 imposed: supply lowers the premium)
    O  log total short OI, detr.  = l_OY ytil + l_Oa alpha + e_O    (l_Oa: demand footprint in open interest)
    D  25-delta skew (demand)      = l_DY ytil + l_Da alpha + e_D
    S  client+pro net short share  = l_SY ytil + l_Sa alpha + e_S
    F  FII net short share         = l_FY ytil + l_Fa alpha + e_F
Models and tests (likelihood ratio):
    M1  {P,O}, l_Oa = 0           vs  M2 {P,O} with l_Oa free       -> demand footprint in open interest
    M2b {P,O}, beta = 0           vs  M2                            -> does demand move supply?
    M2u {P,O}, beta free (no equilibrium restriction) vs M2         -> the equilibrium restriction on beta
    M4  {P,O,D,S,F} free loadings                                   -> which measures load on supply vs demand
Also: profile likelihood in s (chi), implied lead-lag of (P,O) vs the empirical VAR, smoothed shocks
(jump diagnostic), and one-step-ahead out-of-sample forecasts (train to 2024-06-28) against a VAR(1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg import twofactor as TF  # noqa: E402
from vmfg.empirical import realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
DT, RHO = 1.0 / 52.0, 0.05
import os as _o
SPLIT = pd.Timestamp(_o.environ.get("VMFG_SPLIT", "2024-06-28"))

# ------------------------------------------------------------------ data: end-of-week snapshots
india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
d["pi_hat"] = d["IV2"] - RD.har_exante(d, "logs_median", return_forecast=True)
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
ch, _ = U.load_chain_history(ROOT / "data/user/chain_history.csv")
daily = (d[["pi_hat", "lam_back"]].join(poi[["log_short_detr", "seller_net_share", "fii_net_share"]], how="inner")
         .join(ch[["skew_25d", "pcr_oi"]], how="left"))
wk = daily.resample("W-FRI").last().dropna(subset=["pi_hat", "log_short_detr"])
wk = wk.loc[wk["log_short_detr"].first_valid_index():]
import os as _os
OI_END = _os.environ.get("VMFG_OI_END")
TAG = _os.environ.get("VMFG_TAG", "")
if OI_END:
    wk = wk.loc[:OI_END]
COLS = dict(P="pi_hat", O="log_short_detr", D="skew_25d", S="seller_net_share", F="fii_net_share")


def standardise(frame, ref):
    return (frame - ref.mean()) / ref.std()


# ------------------------------------------------------------------ model
class Spec:
    def __init__(self, obs, free_a=None, beta_mode="equilibrium", s_fixed=None):
        self.obs = obs                                   # e.g. ["P", "O"]
        self.free_a = free_a if free_a is not None else {k: True for k in obs if k != "P"}
        self.beta_mode = beta_mode                       # "equilibrium" | "free" | "zero"
        self.s_fixed = s_fixed

    def names(self):
        n = ["log_kappa", "log_kappa_a"]
        if self.beta_mode == "equilibrium" and self.s_fixed is None:
            n.append("logit_s")
        elif self.beta_mode == "free":
            n.append("log_beta")
        n += ["log_r", "log_cP"]
        for k in self.obs:
            if k == "P":
                continue
            n.append(f"l_{k}Y")
            if self.free_a[k]:
                n.append(f"l_{k}a")
        n += [f"log_sig_{k}" for k in self.obs]
        return n

    def unpack(self, th):
        v = dict(zip(self.names(), th))
        kap, kap_a = np.exp(v["log_kappa"]), np.exp(v["log_kappa_a"])
        if self.beta_mode == "equilibrium":
            s = self.s_fixed if self.s_fixed is not None else 1 / (1 + np.exp(-v["logit_s"]))
            beta = TF.beta_of(kap, kap_a, s, RHO)
        elif self.beta_mode == "free":
            s, beta = np.nan, np.exp(v["log_beta"])
        else:
            s, beta = np.nan, 0.0
        r = np.exp(v["log_r"])
        H = []
        for k in self.obs:
            if k == "P":
                cP = np.exp(v["log_cP"])
                H.append([-cP, cP])
            else:
                H.append([v[f"l_{k}Y"], v.get(f"l_{k}a", 0.0) if self.free_a[k] else 0.0])
        R = np.diag([np.exp(2 * v[f"log_sig_{k}"]) for k in self.obs])
        return dict(kappa=kap, kappa_a=kap_a, s=s, beta=beta, r=r, H=np.array(H), R=R)


def system_mats(p):
    M = np.array([[-p["kappa"], p["beta"]], [0.0, -p["kappa_a"]]])
    Sig = solve_continuous_lyapunov(M, -np.diag([1.0, p["r"] ** 2]))
    A = TF.expm_ut(M, DT)
    Q = Sig - A @ Sig @ A.T
    return A, Q, Sig


def kalman(Y, p, smooth=False):
    """Exact Gaussian likelihood by sequential scalar updates (diagonal measurement noise).
    Returns (loglik, dict(pred=one-step-ahead predictions of the observables)); with smooth=True the
    RTS-smoothed states are added (numpy, used once)."""
    A, Q, Sig = system_mats(p)
    H, Rd = p["H"].tolist(), np.diag(p["R"]).tolist()
    a00, a01, a10, a11 = float(A[0, 0]), float(A[0, 1]), float(A[1, 0]), float(A[1, 1])
    q00, q01, q11 = float(Q[0, 0]), float(Q[0, 1]), float(Q[1, 1])
    z0 = z1 = 0.0
    P00, P01, P11 = float(Sig[0, 0]), float(Sig[0, 1]), float(Sig[1, 1])
    L2PI = np.log(2 * np.pi)
    ll = 0.0
    n, m = Y.shape
    rows = Y.tolist()
    pred = np.full((n, m), np.nan)
    keep = smooth
    zp, Pp, zf, Pf = [], [], [], []
    for t in range(n):
        if t > 0:
            z0, z1 = a00 * z0 + a01 * z1, a10 * z0 + a11 * z1
            b00 = a00 * P00 + a01 * P01; b01 = a00 * P01 + a01 * P11
            b10 = a10 * P00 + a11 * P01; b11 = a10 * P01 + a11 * P11
            P00 = b00 * a00 + b01 * a01 + q00
            P01 = b00 * a10 + b01 * a11 + q01
            P11 = b10 * a10 + b11 * a11 + q11
        if keep:
            zp.append((z0, z1)); Pp.append((P00, P01, P11))
        yt = rows[t]
        for j in range(m):
            h0, h1 = H[j]
            pred[t, j] = h0 * z0 + h1 * z1
        for j in range(m):
            y = yt[j]
            if y != y:
                continue
            h0, h1 = H[j]
            Ph0 = P00 * h0 + P01 * h1
            Ph1 = P01 * h0 + P11 * h1
            S = h0 * Ph0 + h1 * Ph1 + Rd[j]
            if S <= 0:
                return -1e12, None
            e = y - (h0 * z0 + h1 * z1)
            ll -= 0.5 * (L2PI + np.log(S) + e * e / S)
            K0, K1 = Ph0 / S, Ph1 / S
            z0 += K0 * e; z1 += K1 * e
            P00 -= K0 * Ph0; P01 -= K0 * Ph1; P11 -= K1 * Ph1
        if keep:
            zf.append((z0, z1)); Pf.append((P00, P01, P11))
    out = dict(pred=pred)
    if smooth:
        mat = lambda c: np.array([[c[0], c[1]], [c[1], c[2]]])
        zs = [None] * n
        zs[-1] = np.array(zf[-1])
        for t in range(n - 2, -1, -1):
            J = mat(Pf[t]) @ A.T @ np.linalg.inv(mat(Pp[t + 1]))
            zs[t] = np.array(zf[t]) + J @ (zs[t + 1] - np.array(zp[t + 1]))
        out["zs"] = np.array(zs)
        out["A"], out["Q"] = A, Q
    return ll, out


def negll(th, spec, Y):
    try:
        p = spec.unpack(th)
        if not (0.05 < p["kappa"] < 500 and 0.05 < p["kappa_a"] < 500):
            return 1e10
        ll, _ = kalman(Y, p)
        return -ll if np.isfinite(ll) else 1e10
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
        return 1e10


def fit(spec, Y, starts=6, seed=0, th0=None):
    rng = np.random.default_rng(seed)
    names = spec.names()
    best = None
    base = dict(log_kappa=np.log(4.0), log_kappa_a=np.log(12.0), logit_s=1.0, log_beta=np.log(0.5), log_r=np.log(3.0),
                log_cP=np.log(0.3))
    for i in range(starts):
        if th0 is not None and i == 0:
            x0 = np.array(th0)
        else:
            x0 = np.array([base.get(nm, 0.0) + (rng.normal(0, 0.6) if i else 0.0) if not nm.startswith(("l_", "log_sig"))
                           else (rng.normal(0, 0.3) if nm.startswith("l_") else np.log(0.6)) for nm in names])
        res = minimize(negll, x0, args=(spec, Y), method="L-BFGS-B", options=dict(maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    return best


def hessian_se(spec, Y, th):
    k = len(th)
    Hm = np.zeros((k, k))
    h = 1e-4
    f0 = negll(th, spec, Y)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            fpp = negll(th + ei + ej, spec, Y); fpm = negll(th + ei - ej, spec, Y)
            fmp = negll(th - ei + ej, spec, Y); fmm = negll(th - ei - ej, spec, Y)
            Hm[i, j] = Hm[j, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    try:
        cov = np.linalg.inv(Hm)
        return np.sqrt(np.clip(np.diag(cov), 0, None)), f0
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), f0


def natural(spec, th, se):
    v = dict(zip(spec.names(), th)); sv = dict(zip(spec.names(), se))
    rows = []
    for nm in spec.names():
        x, sx = v[nm], sv[nm]
        if nm.startswith("log_"):
            rows.append((nm[4:], np.exp(x), np.exp(x) * sx))
        elif nm == "logit_s":
            s = 1 / (1 + np.exp(-x)); rows.append(("s", s, s * (1 - s) * sx))
        else:
            rows.append((nm, x, sx))
    return pd.DataFrame(rows, columns=["parameter", "estimate", "se"])


def lr(ll_u, ll_r, df):
    stat = 2 * (ll_u - ll_r)
    return stat, float(chi2.sf(max(stat, 0.0), df))


# ------------------------------------------------------------------ estimation on the full sample
out = ["# 16 - Latent two-factor state-space estimation", "",
       f"weekly end-of-week snapshots: {len(wk)} weeks, {wk.index[0].date()} to {wk.index[-1].date()}", ""]
Zs = standardise(wk[[COLS[k] for k in "PODSF"]], wk[[COLS[k] for k in "PODSF"]])
Zs.columns = list("PODSF")
Y2 = Zs[["P", "O"]].to_numpy()

M1 = Spec(["P", "O"], free_a={"O": False})
M2 = Spec(["P", "O"], free_a={"O": True})
M2b = Spec(["P", "O"], free_a={"O": True}, beta_mode="zero")
M2u = Spec(["P", "O"], free_a={"O": True}, beta_mode="free")
res = {}
for nm, sp in (("M1", M1), ("M2", M2), ("M2b", M2b), ("M2u", M2u)):
    r = fit(sp, Y2, starts=6, seed=1)
    res[nm] = (sp, r)
ll = {k: -v[1].fun for k, v in res.items()}
t_foot = lr(ll["M2"], ll["M1"], 1)
t_beta0 = lr(ll["M2"], ll["M2b"], 1)
t_eq = lr(ll["M2u"], ll["M2"], 1)
sp2, r2 = res["M2"]
se2, _ = hessian_se(sp2, Y2, r2.x)
nat2 = natural(sp2, r2.x, se2)
p2 = sp2.unpack(r2.x)
pu = res["M2u"][0].unpack(res["M2u"][1].x)
beta_bound = p2["kappa"] * (p2["kappa"] + RHO) / (RHO + p2["kappa"] + p2["kappa_a"])
out += ["## Log-likelihoods", "", pd.Series(ll).round(3).to_string(), "",
        f"- demand footprint in open interest (M1 vs M2, l_Oa = 0): LR = {t_foot[0]:.2f}, p = {t_foot[1]:.4f}",
        f"- demand moves supply (M2b vs M2, beta = 0): LR = {t_beta0[0]:.2f}, p = {t_beta0[1]:.4f}",
        f"- equilibrium restriction on beta (M2 vs M2u): LR = {t_eq[0]:.2f}, p = {t_eq[1]:.4f}; "
        f"free beta = {pu['beta']:.3f} vs equilibrium bound at M2 speeds {beta_bound:.3f}",
        "", "## M2 parameters (natural scale, Hessian s.e.)", "", nat2.round(4).to_string(index=False), ""]

# profile likelihood in s (chi)
prof = []
for sv in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99):
    sp = Spec(["P", "O"], free_a={"O": True}, s_fixed=sv)
    th0 = [x for nm, x in zip(sp2.names(), r2.x) if nm != "logit_s"]
    rr = fit(sp, Y2, starts=4, seed=2, th0=th0)
    prof.append(dict(s=sv, chi_over_b=(1 - sv) / sv, loglik=-rr.fun, LR_vs_max=2 * (ll["M2"] + rr.fun)))
prof = pd.DataFrame(prof)
out += ["## Profile likelihood in s = b/(chi+b) (M2)", "", prof.round(3).to_string(index=False), ""]

# implied lead-lag of (P, O) vs empirical VAR on the same snapshots
def implied_var(p):
    A, Q, Sig = system_mats(p)
    H, R = p["H"], p["R"]
    C0 = H @ Sig @ H.T + R
    C1 = H @ A @ Sig @ H.T
    return C1 @ np.linalg.inv(C0)


Xv = Zs[["P", "O"]].dropna().to_numpy()
cf, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xv) - 1), Xv[:-1]]), Xv[1:], rcond=None)
Phi_emp = cf[1:].T
Phi_M1 = implied_var(res["M1"][0].unpack(res["M1"][1].x))
Phi_M2 = implied_var(p2)
leadlag = pd.DataFrame(dict(empirical=[Phi_emp[0, 1], Phi_emp[1, 0], Phi_emp[0, 0], Phi_emp[1, 1]],
                            M1_no_footprint=[Phi_M1[0, 1], Phi_M1[1, 0], Phi_M1[0, 0], Phi_M1[1, 1]],
                            M2_footprint=[Phi_M2[0, 1], Phi_M2[1, 0], Phi_M2[0, 0], Phi_M2[1, 1]]),
                       index=["O -> future P", "P -> future O", "P persistence", "O persistence"])
out += ["## Lead-lag of the observables (standardised VAR(1) coefficients)", "", leadlag.round(4).to_string(), ""]

# full measurement model: which measures load on supply vs demand
M4 = Spec(list("PODSF"), free_a={k: True for k in "ODSF"})
th0 = list(r2.x[:len([n for n in M2.names() if not n.startswith(("l_", "log_sig"))])])
r4 = fit(M4, Zs[list("PODSF")].to_numpy(), starts=8, seed=3)
se4, _ = hessian_se(M4, Zs[list("PODSF")].to_numpy(), r4.x)
nat4 = natural(M4, r4.x, se4)
p4 = M4.unpack(r4.x)
A4, Q4, S4 = system_mats(p4)
load = []
for i, k in enumerate(M4.obs):
    h = p4["H"][i]
    var_y, var_a = h[0] ** 2 * S4[0, 0], h[1] ** 2 * S4[1, 1]
    cov = 2 * h[0] * h[1] * S4[0, 1]
    tot = h @ S4 @ h + p4["R"][i, i]
    load.append(dict(measure=k, loading_supply=h[0], loading_demand=h[1], share_supply=var_y / tot,
                     share_demand=var_a / tot, share_cross=cov / tot, share_noise=p4["R"][i, i] / tot))
load = pd.DataFrame(load)
out += ["## M4: all five measures, free loadings", "", nat4.round(4).to_string(index=False), "",
        "variance decomposition of each measure:", load.round(3).to_string(index=False), "",
        f"- log-likelihood M4 = {-r4.fun:.3f}", ""]

# smoothed shocks: jump diagnostic (M4)
ll4, sm = kalman(Zs[list("PODSF")].to_numpy(), p4, smooth=True)
zs = sm["zs"]
w = zs[1:] - zs[:-1] @ sm["A"].T
Lq = np.linalg.cholesky(sm["Q"])
eps = np.linalg.solve(Lq, w.T).T
from scipy.stats import kurtosis  # noqa: E402
dates = Zs.index[1:]
top = pd.Series(np.abs(eps[:, 1]), index=dates).sort_values(ascending=False).head(6)
out += ["## Smoothed shocks (M4)", f"- excess kurtosis: supply {kurtosis(eps[:, 0]):.2f}, demand {kurtosis(eps[:, 1]):.2f}",
        "- largest demand shocks (standardised): " + ", ".join(f"{i.date()} ({v:.1f})" for i, v in top.items()), ""]

# out-of-sample: train to SPLIT, one-step-ahead forecasts on the rest
train = wk.loc[:SPLIT]
Zt = standardise(wk[[COLS["P"], COLS["O"]]], train[[COLS["P"], COLS["O"]]])
Ytr, Yall = Zt.loc[:SPLIT].to_numpy(), Zt.to_numpy()
rt = fit(M2, Ytr, starts=8, seed=4)
rt1 = fit(M1, Ytr, starts=8, seed=4)
_, kf2 = kalman(Yall, M2.unpack(rt.x))
_, kf1 = kalman(Yall, M1.unpack(rt1.x))
Xt = Zt.loc[:SPLIT].dropna().to_numpy()
cfv, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Xt) - 1), Xt[:-1]]), Xt[1:], rcond=None)
test_idx = np.where(Zt.index > SPLIT)[0]
rows = []
for j, k in enumerate(["P", "O"]):
    yt = Yall[test_idx, j]
    prev = Yall[test_idx - 1]
    var_f = np.column_stack([np.ones(len(prev)), prev]) @ cfv[:, j]
    ok = np.isfinite(yt) & np.all(np.isfinite(prev), axis=1)
    rmse = lambda f: float(np.sqrt(np.mean((yt[ok] - f[ok]) ** 2)))
    rows.append(dict(series=k, weeks=int(ok.sum()), state_space_M2=rmse(kf2["pred"][test_idx, j]),
                     state_space_M1=rmse(kf1["pred"][test_idx, j]), VAR1=rmse(var_f), no_change=rmse(prev[:, j])))
oos = pd.DataFrame(rows)
out += [f"## Out-of-sample one-step-ahead RMSE (train to {SPLIT.date()}, standardised units)", "",
        oos.round(4).to_string(index=False), ""]

# save
pd.DataFrame(dict(model=list(ll), loglik=list(ll.values()))).to_csv(RES / f"16_loglik{TAG}.csv", index=False)
pd.DataFrame([dict(test="demand footprint in OI (l_Oa=0)", LR=t_foot[0], p=t_foot[1]),
              dict(test="demand moves supply (beta=0)", LR=t_beta0[0], p=t_beta0[1]),
              dict(test="equilibrium restriction on beta", LR=t_eq[0], p=t_eq[1])]).to_csv(RES / f"16_tests{TAG}.csv", index=False)
nat2.to_csv(RES / f"16_M2_params{TAG}.csv", index=False)
prof.to_csv(RES / f"16_profile_s{TAG}.csv", index=False)
leadlag.to_csv(RES / f"16_leadlag{TAG}.csv")
nat4.to_csv(RES / f"16_M4_params{TAG}.csv", index=False)
load.to_csv(RES / f"16_M4_loadings{TAG}.csv", index=False)
oos.to_csv(RES / f"16_oos{TAG}.csv", index=False)
(RES / f"16_meta{TAG}.txt").write_text(f"weeks={len(wk)}\nstart={wk.index[0].date()}\nend={wk.index[-1].date()}\n"
                                 f"kurt_supply={kurtosis(eps[:, 0])}\nkurt_demand={kurtosis(eps[:, 1])}\n"
                                 f"top_demand={';'.join(str(i.date()) for i in top.index)}\nfree_beta={pu['beta']}\n"
                                 f"beta_bound={beta_bound}\n")
(RES / f"16_state_space{TAG}.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
