#!/usr/bin/env python3
"""22 - Natural experiment: SEBI's late-2024 index-derivatives measures as a shock to variance supply.

SEBI circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132 (1 October 2024). Effective 20 November 2024: an extra 2%
extreme-loss margin on SHORT index options on expiry day (a cost borne by sellers), a minimum contract size
of Rs 15 lakh (NIFTY lot 25 -> 75), and one weekly expiry per exchange. Effective 1 February 2025: upfront
collection of the premium from BUYERS and no calendar-spread margin relief on expiry day.

Event 1 (supply-tilted): pre 2024-06-01..2024-09-30 (before the announcement), post 2024-11-22..2025-01-31
(after the effective date, before the buyer-side measures). Event 2 (buyer-side, contrast): pre 2024-11-22..
2025-01-31, post 2025-02-07..2025-03-31.

Supply measure S: client + pro net short share of index-option open interest (scale-free: raw contract counts
are not comparable across the lot-size change). Premium: ex-ante estimate pi_hat (primary), implied variance,
trailing proxy. Global control: the US premium (difference-in-differences). Weekly means (W-FRI).

Identification: with lam = a_t - b Y, an event that lowers supply (dS < 0) with no change in demand gives
b = -d lam / dS (Wald). If the event also lowers demand (larger lots and upfront premium deter buyers), the
Wald ratio is a LOWER bound for b.

Inference: Newey-West s.e. (4 lags) for event effects; stratified block bootstrap for the Wald ratio; placebo
distribution over pseudo-events on the 20th of each month, Nov 2020 - Feb 2024, same window lengths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PROC = ROOT / "results", ROOT / "data" / "processed"
rng = np.random.default_rng(22)


def nw_diff(y, post, lag=4):
    """Effect of Post in y = alpha + delta Post + e, Newey-West s.e."""
    X = np.column_stack([np.ones_like(post, dtype=float), post.astype(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for j in range(1, lag + 1):
        G = Xe[j:].T @ Xe[:-j]
        S += (1 - j / (lag + 1)) * (G + G.T)
    return beta[1], float(np.sqrt((XtXi @ S @ XtXi)[1, 1]))


def build():
    india = pd.read_csv(PROC / "india_daily_github.csv", parse_dates=["date"]).set_index("date")
    d = RD.add_constructions(india)
    d["pi_hat"] = d["IV2"] - RD.har_exante(d, "logs_median", return_forecast=True)
    us = pd.read_csv(PROC / "us_daily_github.csv", parse_dates=["date"]).set_index("date")
    du = RD.add_constructions(us)
    du["pi_hat"] = du["IV2"] - RD.har_exante(du, "logs_median", return_forecast=True)
    poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
    wi = d[["pi_hat", "IV2", "lam_back"]].join(poi[["seller_net_share", "fii_net_share"]], how="inner").resample("W-FRI").mean()
    wu = du[["pi_hat", "IV2", "lam_back"]].resample("W-FRI").mean().add_suffix("_us")
    return wi.join(wu, how="inner").dropna()


def effects(w, pre, post):
    sel_pre = (w.index >= pre[0]) & (w.index <= pre[1])
    sel_post = (w.index >= post[0]) & (w.index <= post[1])
    z = w[sel_pre | sel_post]
    ind = np.asarray((z.index >= post[0]) & (z.index <= post[1])).astype(int)
    out = {}
    for col in ("seller_net_share", "fii_net_share", "pi_hat", "IV2", "lam_back", "pi_hat_us", "IV2_us", "lam_back_us"):
        out[col] = nw_diff(z[col].to_numpy(), ind)
    for col in ("pi_hat", "IV2", "lam_back"):
        out[f"did_{col}"] = nw_diff((z[col] - z[f"{col}_us"]).to_numpy(), ind)
    out["n_pre"], out["n_post"] = int((ind == 0).sum()), int((ind == 1).sum())
    return out, z, ind


def boot_ratio(z, ind, prem="pi_hat", B=4000, L=3):
    pre_idx, post_idx = np.where(ind == 0)[0], np.where(ind == 1)[0]
    S = z["seller_net_share"].to_numpy()
    D = (z[prem] - z[f"{prem}_us"]).to_numpy()

    def blocks(idx):
        n = len(idx)
        nb = -(-n // L)
        st = rng.integers(0, max(n - L + 1, 1), nb)
        return idx[(st[:, None] + np.arange(L)[None, :]).ravel()[:n] % n] if n >= L else idx[rng.integers(0, n, n)]

    reps = []
    for _ in range(B):
        a, b = blocks(pre_idx), blocks(post_idx)
        dS = S[b].mean() - S[a].mean()
        dD = D[b].mean() - D[a].mean()
        reps.append((dS, dD, -dD / dS if dS != 0 else np.nan))
    return np.array(reps)


def main():
    w = build()
    ev1 = (("2024-06-01", "2024-09-30"), ("2024-11-22", "2025-01-31"))
    ev2 = (("2024-11-22", "2025-01-31"), ("2025-02-07", "2025-03-31"))
    lines = ["# 22 - Natural experiment: SEBI measures of 1 October 2024", "",
             f"weekly sample {w.index[0].date()} to {w.index[-1].date()}, {len(w)} weeks (India and US aligned)", ""]
    res = {}
    for name, (pre, post) in (("event 1 (20 Nov 2024, supply-tilted)", ev1), ("event 2 (1 Feb 2025, buyer-side)", ev2)):
        eff, z, ind = effects(w, pre, post)
        res[name] = (eff, z, ind)
        rows = [dict(variable=k, effect=v[0], se=v[1], t=v[0] / v[1]) for k, v in eff.items() if isinstance(v, tuple)]
        lines += [f"## {name}: pre {pre[0]}..{pre[1]} ({eff['n_pre']} weeks), post {post[0]}..{post[1]} ({eff['n_post']} weeks)",
                  pd.DataFrame(rows).round(5).to_string(index=False), ""]
    eff1, z1, ind1 = res["event 1 (20 Nov 2024, supply-tilted)"]
    wald = {}
    for prem in ("pi_hat", "IV2", "lam_back"):
        dS, dD = eff1["seller_net_share"][0], eff1[f"did_{prem}"][0]
        reps = boot_ratio(z1, ind1, prem)
        r = reps[:, 2]
        r = r[np.isfinite(r)]
        wald[prem] = dict(premium=prem, dS=dS, dD=dD, wald_b=-dD / dS, ci_lo=np.percentile(r, 5), ci_hi=np.percentile(r, 95),
                          first_stage_t=eff1["seller_net_share"][0] / eff1["seller_net_share"][1])
    W = pd.DataFrame(wald.values())
    lines += ["## Wald ratio b = -(DiD premium effect)/(supply effect), event 1 (90% stratified block-bootstrap interval)",
              W.round(5).to_string(index=False), ""]
    # placebo distribution
    pl = []
    for ev in pd.date_range("2020-11-20", "2024-02-20", freq="MS") + pd.Timedelta(days=19):
        pre = ((ev - pd.Timedelta(days=172)).strftime("%Y-%m-%d"), (ev - pd.Timedelta(days=51)).strftime("%Y-%m-%d"))
        post = ((ev + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), (ev + pd.Timedelta(days=72)).strftime("%Y-%m-%d"))
        try:
            e, _, _ = effects(w, pre, post)
        except Exception:
            continue
        if e["n_pre"] < 8 or e["n_post"] < 5:
            continue
        pl.append(dict(event=ev.date(), dS=e["seller_net_share"][0], dD=e["did_pi_hat"][0], dFII=e["fii_net_share"][0]))
    P = pd.DataFrame(pl)
    pv = dict(dS=float(np.mean(np.abs(P.dS) >= abs(eff1["seller_net_share"][0]))),
              dD=float(np.mean(np.abs(P.dD) >= abs(eff1["did_pi_hat"][0]))),
              dFII=float(np.mean(np.abs(P.dFII) >= abs(eff1["fii_net_share"][0]))), n=len(P))
    lines += ["## Placebo inference (pseudo-events on the 20th of each month, Nov 2020 - Feb 2024, same windows)",
              f"- placebo p-value, supply effect (seller share): {pv['dS']:.3f}; DiD premium effect: {pv['dD']:.3f}; "
              f"FII share: {pv['dFII']:.3f}; number of placebos {pv['n']}",
              f"- placebo supply effects: 5th {np.percentile(P.dS, 5):.4f}, 95th {np.percentile(P.dS, 95):.4f}; "
              f"DiD premium: 5th {np.percentile(P.dD, 5):.4f}, 95th {np.percentile(P.dD, 95):.4f}", ""]
    rows1 = [dict(variable=k, effect=v[0], se=v[1], t=v[0] / v[1]) for k, v in eff1.items() if isinstance(v, tuple)]
    pd.DataFrame(rows1).to_csv(RES / "22_event1_effects.csv", index=False)
    eff2 = res["event 2 (1 Feb 2025, buyer-side)"][0]
    rows2 = [dict(variable=k, effect=v[0], se=v[1], t=v[0] / v[1]) for k, v in eff2.items() if isinstance(v, tuple)]
    pd.DataFrame(rows2).to_csv(RES / "22_event2_effects.csv", index=False)
    W.to_csv(RES / "22_wald.csv", index=False)
    P.to_csv(RES / "22_placebos.csv", index=False)
    pd.DataFrame([pv]).to_csv(RES / "22_placebo_pvalues.csv", index=False)
    (RES / "22_natural_experiment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
