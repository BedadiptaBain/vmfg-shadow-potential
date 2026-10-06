#!/usr/bin/env python3
"""14b - Lead-lag (VAR) cross effects with three premium measures: trailing proxy, ex-ante estimate, implied variance."""
import sys; from pathlib import Path; ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from vmfg.empirical import inference as F, realdata as RD
from vmfg.marketdata import userfiles as U
from scipy.linalg import expm
india = pd.read_csv(ROOT / "data/processed/india_daily_github.csv", parse_dates=["date"]).set_index("date")
d = RD.add_constructions(india)
Fi = RD.har_exante(d, "logs_median", return_forecast=True)
d["pi_hat"] = d["IV2"] - Fi
poi, _ = U.load_participant_oi(ROOT / "data/user/participant_oi.csv")
rng = np.random.default_rng(15)
rows = []
for col, lab in (("lam_back", "trailing proxy"), ("pi_hat", "ex-ante estimate"), ("IV2", "implied variance")):
    wk = d[[col]].join(poi[["log_short_detr"]], how="inner").resample("W-FRI").mean().dropna()
    import os as _os
    if _os.environ.get("VMFG_OI_END"):
        wk = wk.loc[:_os.environ["VMFG_OI_END"]]
    W = wk.to_numpy(); n = len(W)
    def coef(A0, A1):
        X0 = np.column_stack([np.ones(len(A0)), A0]); c, *_ = np.linalg.lstsq(X0, A1, rcond=None); return c[1:].T
    Phi = coef(W[:-1], W[1:])
    reps = np.array([coef(W[:-1][ib], W[1:][ib]) for ib in (F.block_indices(n - 1, 8, rng) for _ in range(2000))])
    se = reps.std(axis=0, ddof=1)
    rows.append(dict(premium=lab, weeks=n, supply_to_premium=Phi[0, 1], t_s2p=Phi[0, 1] / se[0, 1],
                     premium_to_supply=Phi[1, 0], t_p2s=Phi[1, 0] / se[1, 0],
                     persistence_premium=Phi[0, 0], persistence_supply=Phi[1, 1]))
out = pd.DataFrame(rows); out.to_csv(ROOT / ("results/14_V1_robustness" + __import__("os").environ.get("VMFG_TAG", "") + ".csv"), index=False)
print(out.round(4).to_string(index=False))
