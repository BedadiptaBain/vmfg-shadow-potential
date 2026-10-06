"""Loaders for the author's own platform exports (Apps Script / Kite pipeline).

participant_oi.csv  NSE participant-wise INDEX OPTION open interest (contracts):
    client/dii/fii/pro x long/short, plus derived net-short columns.
chain_history.csv   daily NIFTY option-chain summary for the nearest expiry:
    spot, dte, atm_iv (%), total_OI, vega_outstanding_per_volpt (Rs per vol point),
    net_dealer_gamma, skew_25d, pcr_oi and concentration measures.

Derived deployment proxies (candidates for the model's aggregate state Y):
    short_total        total short index-option OI (= total long OI)
    seller_net_share   (net short client + net short pro) / short_total
    fii_net_share      net short FII / short_total
    log_short_detr     log short_total minus its 250-session rolling mean
    log_vega_detr      log vega outstanding minus its 250-session rolling mean
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

POI_REQUIRED = ["date", "client_long", "client_short", "dii_long", "dii_short", "fii_long", "fii_short",
                "pro_long", "pro_short", "net_short_pro", "net_short_fii", "net_short_client"]
CHAIN_REQUIRED = ["date", "expiry", "dte", "spot", "total_OI", "atm_iv", "vega_outstanding_per_volpt"]


def _load(path: Path, required: list[str]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"])
    audit = dict(file=path.name, rows=len(df), first=str(df["date"].min().date()), last=str(df["date"].max().date()),
                 duplicates=int(df["date"].duplicated().sum()), was_sorted=bool(df["date"].is_monotonic_increasing),
                 nan_cells=int(df.isna().sum().sum()))
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    return df, audit


def load_participant_oi(path: Path) -> tuple[pd.DataFrame, dict]:
    df, audit = _load(Path(path), POI_REQUIRED)
    longs = df[["client_long", "dii_long", "fii_long", "pro_long"]].sum(axis=1)
    shorts = df[["client_short", "dii_short", "fii_short", "pro_short"]].sum(axis=1)
    audit["long_equals_short_share"] = float(np.mean(np.isclose(longs, shorts, rtol=1e-6)))
    df["short_total"] = shorts
    df["seller_net_share"] = (df["net_short_client"] + df["net_short_pro"]) / shorts
    df["fii_net_share"] = df["net_short_fii"] / shorts
    ls = np.log(shorts)
    df["log_short_detr"] = ls - ls.rolling(250, min_periods=120).mean()
    return df, audit


def load_chain_history(path: Path) -> tuple[pd.DataFrame, dict]:
    df, audit = _load(Path(path), CHAIN_REQUIRED)
    lv = np.log(df["vega_outstanding_per_volpt"].clip(lower=1.0))
    df["log_vega_detr"] = lv - lv.rolling(250, min_periods=120).mean()
    df["atm_iv2"] = (df["atm_iv"] / 100.0) ** 2
    audit["atm_iv_range"] = (float(df["atm_iv"].min()), float(df["atm_iv"].max()))
    audit["dte_range"] = (float(df["dte"].min()), float(df["dte"].max()))
    return df, audit
