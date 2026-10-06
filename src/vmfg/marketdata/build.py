"""Turn raw downloads into canonical daily datasets (documented rules only).

India (source "github": one-minute NIFTY 50 bars + Yahoo India VIX)
  R1 regular hours      keep bars with time in [09:15, 15:30]
  R2 sessions           keep dates with >= 120 regular-hours bars that are
                        weekdays, plus the whitelisted real weekend sessions
                        SPECIAL_LIVE_SESSIONS; this removes NSE mock-trading
                        Saturdays and Diwali muhurat sessions
  R3 daily close        mean of the one-minute closes 15:00-15:29, a proxy for
                        NSE's official close (a 15:00-15:30 average); checked
                        against Yahoo's official closes (2020-2026)
  R4 returns            r_t = log(C_t / C_{t-1}) on the R2 session calendar
  R5 intraday RV        rv5_t = sum of squared 5-minute log returns
                        (09:15 open, then 5-minute bucket closes); overnight
                        return r_on_t = log(Open_t / C_{t-1}); both daily,
                        not annualised
  R6 implied vol        India VIX close from Yahoo (two pinned downloads,
                        which must agree); no filling of missing days
  R7 merge              left-join VIX onto the R2 calendar, so returns never
                        span a missing VIX day

US (source "github": SPY + CBOE VIX)
  close = SPY adjusted close (dividends/splits), vix = CBOE VIX close,
  calendar = SPY trading days.

Output columns: date, close, r, vix, rv5, r_on, n_bars, r_official
(r_official = Yahoo official-close log return where available).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPECIAL_LIVE_SESSIONS = pd.to_datetime(
    ["2020-02-01", "2024-01-20", "2024-03-02", "2024-05-18", "2025-02-01", "2026-02-01"])
MIN_BARS = 120


# ---------------------------------------------------------------------------
# readers for raw files
# ---------------------------------------------------------------------------
def read_minute_bars(raw_root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw_root / "github" / "nifty50_1min" / "1min" / "*" / "*.csv")))
    if not files:
        raise FileNotFoundError("no NIFTY one-minute files; run scripts/00_download_data.py --source github")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["Timestamp"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["date"] = df["ts"].dt.normalize()
    df["hm"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df


def read_yahoo_vix_mirrors(raw_root: Path) -> tuple[pd.Series, dict]:
    a = pd.read_csv(raw_root / "github/indiavix_yahoo/data/raw/india_vix.csv", parse_dates=["date"])
    a = a.set_index("date")["vix_close"].astype(float)
    p = raw_root / "github/nifty_vix_yahoo_2020/outputs/forecasts/garch_vix_comparison.csv"
    b = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["India_VIX"].astype(float)
    ov = a.index.intersection(b.index)
    audit = dict(vix_a_rows=len(a), vix_b_rows=len(b), overlap=len(ov),
                 max_abs_diff_on_overlap=float((a.loc[ov] - b.loc[ov]).abs().max()))
    vix = pd.concat([a, b[~b.index.isin(a.index)]]).sort_index()
    vix.index.name = "date"
    return vix, audit


def read_yahoo_official_returns(raw_root: Path) -> pd.Series:
    p = raw_root / "github/nifty_vix_yahoo_2020/outputs/forecasts/test_volatility_forecasts.csv"
    r = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Actual_Return"] / 100.0
    r.index.name = "date"
    return r


# ---------------------------------------------------------------------------
# India from one-minute bars
# ---------------------------------------------------------------------------
def daily_from_minutes(bars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reg = bars[(bars["hm"] >= 555) & (bars["hm"] <= 930)]                     # R1
    n = reg.groupby("date").size()
    all_days = n.index
    is_wl = all_days.isin(SPECIAL_LIVE_SESSIONS)
    keep = all_days[((n.values >= MIN_BARS) & (all_days.dayofweek < 5)) | is_wl]   # R2
    dropped = sorted(set(bars["date"].unique()) - set(keep))
    reg = reg[reg["date"].isin(keep)]
    g = reg.groupby("date")
    close_window = reg[(reg["hm"] >= 900) & (reg["hm"] <= 929)].groupby("date")["Close"].mean()
    close_last = g["Close"].last()
    close = close_window.reindex(keep).fillna(close_last)                      # R3 (short sessions: last bar)
    first_open = g["Open"].first()
    # R5: 5-minute bucket closes, starting from the session open
    reg = reg.assign(bucket=(reg["hm"] - 555) // 5)
    bc = reg.groupby(["date", "bucket"])["Close"].last()
    lp = np.log(bc)
    r5 = lp.groupby(level=0).diff()
    first_b = lp.groupby(level=0).head(1)
    r5.loc[first_b.index] = first_b.values - np.log(first_open.reindex(first_b.index.get_level_values(0))).values
    rv5 = (r5 ** 2).groupby(level=0).sum()
    out = pd.DataFrame({"close": close, "open": first_open, "n_bars": n.reindex(keep), "rv5": rv5})
    out.index.name = "date"
    out = out.sort_index()
    out["r"] = np.log(out["close"]).diff()                                      # R4
    out["r_on"] = np.log(out["open"] / out["close"].shift(1))
    audit = dict(minute_bars=int(len(bars)), feed_days=int(bars["date"].nunique()), sessions_kept=int(len(keep)),
                 dropped_days=[d.strftime("%Y-%m-%d") for d in dropped],
                 special_sessions_kept=[d.strftime("%Y-%m-%d") for d in keep[keep.isin(SPECIAL_LIVE_SESSIONS)]])
    return out, audit


def build_india_github(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    bars = read_minute_bars(raw_root)
    daily, audit = daily_from_minutes(bars)
    vix, vaudit = read_yahoo_vix_mirrors(raw_root)
    daily = daily.join(vix.rename("vix"), how="left")                           # R7
    off = read_yahoo_official_returns(raw_root)
    # official returns are only comparable when both calendars share the previous session
    prev_ours = pd.Series(daily.index, index=daily.index).shift(1)
    prev_off = pd.Series(off.index, index=off.index).shift(1)
    same_prev = [d for d in daily.index if d in off.index and prev_ours.get(d) == prev_off.get(d)]
    daily["r_official"] = np.nan
    daily.loc[same_prev, "r_official"] = off.loc[same_prev]
    diff = (daily["r"] - daily["r_official"]).dropna()
    worst = diff.abs().sort_values(ascending=False).head(5)
    audit.update(vaudit)
    audit.update(
        first_date=daily.index[0].strftime("%Y-%m-%d"), last_date=daily.index[-1].strftime("%Y-%m-%d"),
        sessions_missing_vix=int(daily["vix"].isna().sum()),
        missing_vix_dates=[d.strftime("%Y-%m-%d") for d in daily.index[daily["vix"].isna()] if d.year >= 2018],
        vix_dates_not_sessions=[d.strftime("%Y-%m-%d") for d in vix.index if d >= daily.index[0]
                                and d <= daily.index[-1] and d not in daily.index],
        official_check_n=int(len(diff)), official_check_corr=float(np.corrcoef(
            daily["r"].loc[diff.index], daily["r_official"].loc[diff.index])[0, 1]),
        official_check_rms_bp=float(diff.std() * 1e4),
        official_check_worst=[(d.strftime("%Y-%m-%d"), round(float(x) * 1e4, 1)) for d, x in worst.items()],
        sessions_per_year={int(k): int(v) for k, v in daily.groupby(daily.index.year).size().items()},
    )
    return daily.reset_index(), audit


# ---------------------------------------------------------------------------
# US: SPY + CBOE VIX
# ---------------------------------------------------------------------------
def build_us_github(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    spy = pd.read_csv(raw_root / "github/spy_yahoo/data/SPY_Yahoo_yfinance.csv")
    spy["date"] = pd.to_datetime(spy["Date"].str.slice(0, 10))
    spy = spy.set_index("date").sort_index()
    vix = pd.read_csv(raw_root / "github/vix_cboe/data/vix-daily.csv")
    vix["date"] = pd.to_datetime(vix["DATE"])
    vix = vix.set_index("date")["CLOSE"].astype(float)
    out = pd.DataFrame({"close": spy["Adj Close"].astype(float), "close_raw": spy["Close"].astype(float)})
    out["r"] = np.log(out["close"]).diff()
    out = out.join(vix.rename("vix"), how="left")
    audit = dict(first_date=out.index[0].strftime("%Y-%m-%d"), last_date=out.index[-1].strftime("%Y-%m-%d"),
                 sessions=int(len(out)), sessions_missing_vix=int(out["vix"].isna().sum()))
    return out.reset_index(), audit


# ---------------------------------------------------------------------------
# official-source builders (run on your own machine)
# ---------------------------------------------------------------------------
def _read_yahoo_csv(p: Path, col: str = "Close") -> pd.Series:
    df = pd.read_csv(p)
    date_col = [c for c in df.columns if c.lower() in ("date", "datetime", "price")][0]
    df = df[pd.to_datetime(df[date_col], errors="coerce").notna()]
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.normalize()
    return s.dropna().sort_index()


def build_from_yahoo(raw_root: Path, market: str) -> tuple[pd.DataFrame, dict]:
    d = raw_root / "yahoo" / market
    if market == "india":
        close, vix = _read_yahoo_csv(d / "NSEI.csv"), _read_yahoo_csv(d / "INDIAVIX.csv")
    else:
        close, vix = _read_yahoo_csv(d / "SPY.csv", "Adj Close"), _read_yahoo_csv(d / "VIX.csv")
    out = pd.DataFrame({"close": close})
    out["r"] = np.log(out["close"]).diff()
    out = out.join(vix.rename("vix"), how="left")
    out.index.name = "date"
    return out.reset_index(), dict(first_date=str(out.index[0].date()), last_date=str(out.index[-1].date()),
                                   sessions=len(out), sessions_missing_vix=int(out["vix"].isna().sum()))


def _nse_records(obj) -> list[dict]:
    """Find the list of record dicts inside an NSE JSON payload."""
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            rec = _nse_records(v)
            if rec:
                return rec
    return []


def _pick(rec: dict, *keys):
    for k in rec:
        if any(key in k.upper() for key in keys):
            return rec[k]
    return None


def build_from_nse(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    def load(prefix):
        rows = []
        for f in sorted((raw_root / "nse").glob(f"{prefix}_*.json")):
            for rec in _nse_records(json.loads(f.read_text())):
                d = _pick(rec, "TIMESTAMP", "DATE")
                c = _pick(rec, "CLOSE_INDEX_VAL", "CLOSE")
                if d is not None and c is not None:
                    rows.append((pd.to_datetime(d, dayfirst=True).normalize(), float(str(c).replace(",", ""))))
        s = pd.Series(dict(rows)).sort_index()
        if s.empty:
            raise RuntimeError(f"no {prefix} records parsed from data/raw/nse; the NSE payload format may have changed")
        return s
    close, vix = load("nifty50"), load("vix")
    out = pd.DataFrame({"close": close})
    out["r"] = np.log(out["close"]).diff()
    out = out.join(vix.rename("vix"), how="left")
    out.index.name = "date"
    return out.reset_index(), dict(first_date=str(out.index[0].date()), last_date=str(out.index[-1].date()),
                                   sessions=len(out), sessions_missing_vix=int(out["vix"].isna().sum()))


def build_from_kite(raw_root: Path) -> tuple[pd.DataFrame, dict]:
    def load(tag):
        rows = []
        for f in sorted((raw_root / "kite" / "day").glob(f"{tag}_*.json")):
            for c in json.loads(f.read_text())["data"]["candles"]:
                rows.append((pd.to_datetime(c[0]).tz_localize(None).normalize(), float(c[4])))
        return pd.Series(dict(rows)).sort_index()
    close, vix = load("nifty50"), load("indiavix")
    out = pd.DataFrame({"close": close})
    out["r"] = np.log(out["close"]).diff()
    out = out.join(vix.rename("vix"), how="left")
    out.index.name = "date"
    return out.reset_index(), dict(first_date=str(out.index[0].date()), last_date=str(out.index[-1].date()),
                                   sessions=len(out), sessions_missing_vix=int(out["vix"].isna().sum()))
