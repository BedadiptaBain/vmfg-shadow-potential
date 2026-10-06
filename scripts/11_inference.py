#!/usr/bin/env python3
"""11 - Standard errors and tests for every empirical number in Section 11.

Methods (src/vmfg/empirical/inference.py): moving-block bootstrap of observations
and of consecutive valid pairs (L = 63 sessions, B = 2000; sensitivity L = 21, 126),
Newey-West s.e. for means (42 lags) and regressions (25 lags), ADF with 21 lags,
delta method for speeds, Monte Carlo p-values against the fitted OU (R = 2000).
Gaps (missing VIX) are treated as unobserved sessions ("break").
Writes results/11_inference.md, results/11_*.csv and paper/tab11_*.tex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import diagnostics as G, identification as I, inference as F, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PAPER, PROC = ROOT / "results", ROOT / "paper", ROOT / "data" / "processed"
H = G.PAPER_THRESHOLDS
END = pd.Timestamp("2026-08-26")
B, L0 = 2000, 63


def nw_regression_pairs(x: np.ndarray, lag: int = 25) -> pd.DataFrame:
    """Drift regression on valid consecutive pairs only: d = x_{t+1} - x_t on
    [1, x_t, x_t^2, min(x_t, 0)], Newey-West s.e."""
    x0, x1 = F.valid_pairs(x)
    y = x1 - x0
    X = np.column_stack([np.ones_like(x0), x0, x0 ** 2, np.minimum(x0, 0.0)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lag + 1):
        Gm = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lag + 1)) * (Gm + Gm.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return pd.DataFrame(dict(coef=beta, se=se, t=beta / se), index=["const", "lam", "lam^2", "min(lam,0)"])


def ou_benchmark(st: dict, R: int = 2000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    n, m, sd, phi = st["n"], st["mean"], st["sd"], min(max(st["phi"], 0.0), 0.9999)
    x = np.empty((R, n))
    x[:, 0] = rng.standard_normal(R)
    e = rng.standard_normal((R, n)) * np.sqrt(1 - phi ** 2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    x = m + sd * x
    mins = x.min(axis=1)
    shares = (x <= 0).mean(axis=1)
    eps = np.array([len(G.runs_nonpositive(r)) for r in x])
    longest = np.array([max(G.runs_nonpositive(r) or [0]) for r in x])
    return dict(p_min=F.mc_pvalue_lower(mins, st["min"]), min_band=np.percentile(mins, [2.5, 97.5]),
                p_share=F.mc_pvalue_two_sided(shares, st["frac_nonpos"]), share_band=np.percentile(shares, [2.5, 97.5]),
                p_episodes=F.mc_pvalue_two_sided(eps, st["episodes"]), episodes_band=np.percentile(eps, [2.5, 97.5]),
                p_longest=F.mc_pvalue_two_sided(longest, st["longest"]), longest_band=np.percentile(longest, [2.5, 97.5]), R=R)


def analyse(name: str, lam: pd.Series, mean_lag: int = 42) -> dict:
    st = RD.stats(lam, gaps="break")
    x = lam.to_numpy(dtype=float)
    bs = F.bootstrap_series(x, H, L=L0, B=B, seed=11)
    sens = {Lx: F.bootstrap_series(x, H, L=Lx, B=600, seed=12)["se"] for Lx in (21, 126)}
    mean, se_mean = F.nw_mean(x, mean_lag)
    ad = F.adf(x, p=21)
    reg = nw_regression_pairs(x)
    kap, se_kap = F.kappa_delta(bs["est"]["phi"], bs["se"]["phi"], 252.0)
    ou = ou_benchmark(st)
    return dict(name=name, st=st, bs=bs, sens=sens, mean=mean, se_mean=se_mean, adf=ad, reg=reg,
                kappa=kap, se_kappa=se_kap, ou=ou)


def main() -> None:
    india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
    d = RD.add_constructions(india)
    do = RD.add_constructions(india, use_official=True)
    us = pd.read_csv(PROC / "us_daily_github.csv", parse_dates=["date"]).set_index("date")
    du = RD.add_constructions(us)
    lb = d["lam_back"].loc[:END]
    valid = lb.dropna()
    r1_span = (valid.index[-1964], valid.index[-1])
    samples = {"India R1": lb.loc[r1_span[0]:r1_span[1]],
               "India R4": lb.loc["2020-01-01":END],
               "India R5 (official)": do["lam_back"].loc["2020-01-01":END],
               "US 2005-2026": du["lam_back"],
               "US 2020-2026": du["lam_back"].loc["2020-01-01":]}
    out = {k: analyse(k, v) for k, v in samples.items()}

    # the pre-audit convention, for the record
    close = RD.stats(lb.loc[r1_span[0]:r1_span[1]], gaps="close")
    brk = out["India R1"]["st"]
    gap_days = int(lb.loc[r1_span[0]:r1_span[1]].isna().sum())

    # ---------- ex-ante premium: shares and medians with block-bootstrap s.e.
    ex_rows = []
    for mk, dd, start in (("NIFTY", d, "2020-01-01"), ("US", du, "2008-01-01")):
        for v in ("levels", "logs", "logs_median", "levels_iv"):
            F_ = RD.har_exante(dd, v, return_forecast=True)
            e = (dd["RVf"] - F_).loc[start:].dropna()
            pi = (dd["IV2"] - F_).loc[e.index]
            sh, se_sh = F.bootstrap_mean_indicator((pi <= 0).astype(float).to_numpy(), L0, B, seed=5)
            md, se_md = F.bootstrap_median(e.to_numpy(), L0, B, seed=6)
            below, se_below = F.bootstrap_mean_indicator((e < 0).astype(float).to_numpy(), L0, B, seed=7)
            ex_rows.append(dict(market=mk, forecast=v, n=len(e), share=sh, se_share=se_sh, median_error=md,
                                se_median=se_md, share_rv_below=below, se_below=se_below,
                                oos_r2=1 - np.mean(e ** 2) / np.var(dd["RVf"].loc[e.index])))
    ex = pd.DataFrame(ex_rows)

    # ---------- mechanism with block-bootstrap s.e. (weekly, L = 8 weeks)
    poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
    ch, _ = U.load_chain_history(ROOT / "data/user/chain_history.csv")
    m = d[["IV2", "lam_back"]].join(poi[["log_short_detr", "seller_net_share", "fii_net_share"]], how="inner") \
                              .join(ch[["log_vega_detr"]], how="left")
    wk = m.resample("W-FRI").mean().dropna()
    rng = np.random.default_rng(21)
    proxies = ["log_short_detr", "seller_net_share", "fii_net_share", "log_vega_detr"]
    mech_rows = []
    W = wk.to_numpy()
    cols = list(wk.columns)
    dW = np.diff(W, axis=0)
    idx_b = [F.block_indices(len(W), 8, rng) for _ in range(B)]
    idx_d = [F.block_indices(len(dW), 8, rng) for _ in range(B)]
    for pr in ("IV2", "lam_back"):
        for px in proxies:
            i, j = cols.index(pr), cols.index(px)
            c_lv = np.corrcoef(W[:, i], W[:, j])[0, 1]
            c_ch = np.corrcoef(dW[:, i], dW[:, j])[0, 1]
            se_lv = np.std([np.corrcoef(W[ib, i], W[ib, j])[0, 1] for ib in idx_b], ddof=1)
            se_ch = np.std([np.corrcoef(dW[ib, i], dW[ib, j])[0, 1] for ib in idx_d], ddof=1)
            mech_rows.append(dict(premium=pr, proxy=px, corr_levels=c_lv, se_levels=se_lv,
                                  z_vs_minus1=(c_lv + 1) / se_lv, corr_changes=c_ch, se_changes=se_ch))
    mech = pd.DataFrame(mech_rows)
    sp_rows = []
    pairs = {c: F.valid_pairs(wk[c].to_numpy()) for c in ["lam_back"] + proxies}
    npair = len(pairs["lam_back"][0])
    idx_p = [F.block_indices(npair, 8, rng) for _ in range(B)]
    for c in ["lam_back"] + proxies:
        x0, x1 = pairs[c]
        ph = F.phi_hat(x0, x1)
        reps = np.array([F.phi_hat(x0[ib], x1[ib]) for ib in idx_p])
        kap, se_kap = F.kappa_delta(ph, np.std(reps, ddof=1), 52.0)
        if c != "lam_back":
            y0, y1 = pairs["lam_back"]
            diff = np.array([-52 * np.log(F.phi_hat(y0[ib], y1[ib])) + 52 * np.log(F.phi_hat(x0[ib], x1[ib]))
                             for ib in idx_p])
            dk, se_dk = (-52 * np.log(F.phi_hat(y0, y1)) - kap), np.std(diff, ddof=1)
        else:
            dk, se_dk = np.nan, np.nan
        sp_rows.append(dict(series=c, phi=ph, kappa=kap, se_kappa=se_kap, diff_vs_premium=dk, se_diff=se_dk))
    speeds = pd.DataFrame(sp_rows)

    # ---------- identification Monte Carlo with Monte Carlo s.e.
    panel = I.simulate_panel(R=400, premium="reflected", seed=2027)
    ident = {}
    for series in ("pi", "lam_back", "lam_fwd"):
        X = panel[series]
        shares = (X <= 0).mean(axis=1)
        mins = X.min(axis=1)
        rr = np.random.default_rng(3)
        med_se = lambda v: float(np.std([np.median(v[rr.integers(0, len(v), len(v))]) for _ in range(1000)], ddof=1))
        ident[series] = dict(share_q=np.percentile(shares, [5, 50, 95]), share_med_se=med_se(shares),
                             min_q=np.percentile(mins, [5, 50, 95]), min_med_se=med_se(mins),
                             share_mean=shares.mean(), share_mean_se=shares.std(ddof=1) / np.sqrt(len(shares)))

    # ---------- write
    RES.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    rows = []
    for k, o in out.items():
        st, bs = o["st"], o["bs"]
        rows.append(dict(sample=k, n=st["n"], first=st["first"], last=st["last"], gap_days=int(samples[k].isna().sum()),
                         share=bs["est"]["share"], se_share=bs["se"]["share"],
                         se_share_L21=o["sens"][21]["share"], se_share_L126=o["sens"][126]["share"],
                         p_share_ou=o["ou"]["p_share"], share_band_lo=o["ou"]["share_band"][0], share_band_hi=o["ou"]["share_band"][1],
                         episodes=st["episodes"], p_episodes=o["ou"]["p_episodes"],
                         episodes_band_lo=o["ou"]["episodes_band"][0], episodes_band_hi=o["ou"]["episodes_band"][1],
                         longest_band_lo=o["ou"]["longest_band"][0], longest_band_hi=o["ou"]["longest_band"][1],
                         min_band_lo=o["ou"]["min_band"][0], min_band_hi=o["ou"]["min_band"][1],
                         longest=st["longest"], p_longest=o["ou"]["p_longest"],
                         min=st["min"], min_date=st["min_date"], p_min=o["ou"]["p_min"],
                         mean=o["mean"], se_mean=o["se_mean"], skew=bs["est"]["skew"], se_skew=bs["se"]["skew"],
                         skew_lo=bs["lo"]["skew"], skew_hi=bs["hi"]["skew"],
                         exkurt=bs["est"]["exkurt"], se_exkurt=bs["se"]["exkurt"],
                         gauss_share=bs["est"]["gauss_share"], gauss_gap=bs["est"]["gauss_gap"], se_gauss_gap=bs["se"]["gauss_gap"],
                         phi=bs["est"]["phi"], se_phi=bs["se"]["phi"], kappa=o["kappa"], se_kappa=o["se_kappa"],
                         adf_t=o["adf"]["t"], t_lam2=o["reg"].loc["lam^2", "t"], t_hinge=o["reg"].loc["min(lam,0)", "t"],
                         **{f"n_{h}": bs["est"][f"n_{h}"] for h in H}, **{f"p_{h}": bs["est"][f"p_{h}"] for h in H},
                         **{f"se_p_{h}": bs["se"][f"p_{h}"] for h in H}, **{f"se_n_{h}": bs["se"][f"n_{h}"] for h in H}))
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "11_inference_samples.csv", index=False)
    ex.to_csv(RES / "11_inference_exante.csv", index=False)
    mech.to_csv(RES / "11_inference_mechanism.csv", index=False)
    speeds.to_csv(RES / "11_inference_speeds.csv", index=False)
    audit = dict(r1_span=[str(r1_span[0].date()), str(r1_span[1].date())], r1_gap_days=gap_days,
                 close=dict(episodes=close["episodes"], longest=close["longest"], n=list(close["table"]["n"]),
                            p=[round(float(v), 3) for v in close["table"]["p"]]),
                 brk=dict(episodes=brk["episodes"], longest=brk["longest"], n=list(brk["table"]["n"]),
                          p=[round(float(v), 3) for v in brk["table"]["p"]]),
                 ident={k: {kk: (np.round(vv, 4).tolist() if hasattr(vv, "__len__") else round(float(vv), 4))
                            for kk, vv in v.items()} for k, v in ident.items()})
    (RES / "11_inference_audit.json").write_text(json.dumps(audit, indent=1))
    pd.set_option("display.width", 250)
    report = ["# 11 - Inference for Section 11", "", "## Samples", "", tab.round(4).T.to_string(), "",
              "## Ex-ante premium", "", ex.round(4).to_string(), "", "## Mechanism", "", mech.round(3).to_string(), "",
              speeds.round(3).to_string(), "", "## Audit", "", json.dumps(audit, indent=1)]
    (RES / "11_inference.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
