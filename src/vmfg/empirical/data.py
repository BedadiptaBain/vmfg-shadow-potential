"""Load and validate daily NIFTY + India VIX data.

Accepted inputs
  * CSV or XLSX with a date column and two numeric columns:
      date  : one of  date, timestamp, trade_date            (day-first parsing)
      close : one of  nifty, close, nifty_close, nifty50, nifty_50, price
      vix   : one of  vix, india_vix, indiavix, iv, implied_vol
    This matches the DAILY sheet of the Apps Script research workbook
    (columns date, nifty, vix) and NSE / niftyindices.com downloads once
    the two files are merged on the date.
  * Two NSE downloads (index history + India VIX history): use
    load_two_nse_files(index_csv, vix_csv).

Validation (every check is reported, none is silent)
  * rows with unparseable dates or non-positive prices are dropped
  * duplicated dates are dropped (first kept)
  * the input order is recorded and the frame is SORTED by date: a rolling
    realised-variance window on unsorted rows is meaningless
  * VIX units: if the median is below 1.5 the column is taken to be a
    decimal volatility and multiplied by 100
  * daily log returns with |r| > 0.15 and calendar gaps > 7 days are flagged
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATE_KEYS = ("date", "timestamp", "trade_date")
CLOSE_KEYS = ("nifty", "close", "nifty_close", "nifty50", "nifty_50", "price")
VIX_KEYS = ("vix", "india_vix", "indiavix", "iv", "implied_vol")


@dataclass
class ValidationReport:
    n_input_rows: int = 0
    n_dropped_bad: int = 0
    n_duplicates: int = 0
    input_was_sorted: bool = True
    vix_rescaled: bool = False
    big_moves: list = field(default_factory=list)
    calendar_gaps: list = field(default_factory=list)
    first_date: str = ""
    last_date: str = ""
    n_sessions: int = 0

    def as_markdown(self) -> str:
        lines = [
            f"- input rows: {self.n_input_rows}; dropped (bad date/price): {self.n_dropped_bad}; "
            f"duplicate dates dropped: {self.n_duplicates}",
            f"- input already sorted by date: **{self.input_was_sorted}** (the frame is sorted regardless)",
            f"- VIX rescaled from decimals to percent: {self.vix_rescaled}",
            f"- sample: {self.first_date} to {self.last_date}, {self.n_sessions} sessions",
            f"- daily |log return| > 15%: {len(self.big_moves)} "
            + (str(self.big_moves[:10]) if self.big_moves else ""),
            f"- calendar gaps > 7 days: {len(self.calendar_gaps)} "
            + (str(self.calendar_gaps[:10]) if self.calendar_gaps else ""),
        ]
        return "\n".join(lines)


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace(".", "")


def _pick(cols: dict, keys) -> str:
    for k in keys:
        if k in cols:
            return cols[k]
    raise KeyError(f"none of the columns {keys} found; available: {list(cols.values())}")


def _read_any(path: str | Path, sheet: str | None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    return pd.to_datetime(series.astype(str).str.strip(), dayfirst=True, errors="coerce", format="mixed")


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport(n_input_rows=len(df))
    out = df.copy()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    bad = out["date"].isna() | ~(out["close"] > 0) | ~(out["vix"] > 0)
    rep.n_dropped_bad = int(bad.sum())
    out = out.loc[~bad]
    dup = out["date"].duplicated(keep="first")
    rep.n_duplicates = int(dup.sum())
    out = out.loc[~dup]
    rep.input_was_sorted = bool(out["date"].is_monotonic_increasing)
    out = out.sort_values("date").reset_index(drop=True)
    if out["vix"].median() < 1.5:
        out["vix"] = 100.0 * out["vix"]
        rep.vix_rescaled = True
    r = np.log(out["close"]).diff()
    rep.big_moves = [(d.strftime("%Y-%m-%d"), round(float(x), 4))
                     for d, x in zip(out["date"], r) if np.isfinite(x) and abs(x) > 0.15]
    gaps = out["date"].diff().dt.days
    rep.calendar_gaps = [(d.strftime("%Y-%m-%d"), int(g))
                         for d, g in zip(out["date"], gaps) if np.isfinite(g) and g > 7]
    if len(out):
        rep.first_date = out["date"].iloc[0].strftime("%Y-%m-%d")
        rep.last_date = out["date"].iloc[-1].strftime("%Y-%m-%d")
    rep.n_sessions = len(out)
    return out[["date", "close", "vix"]], rep


def load_daily(path: str | Path, sheet: str | None = "DAILY") -> tuple[pd.DataFrame, ValidationReport]:
    """Load one file holding date, NIFTY close and India VIX."""
    try:
        raw = _read_any(path, sheet)
    except ValueError:  # sheet name not present
        raw = _read_any(path, None)
    cols = {_norm(c): c for c in raw.columns}
    df = pd.DataFrame({
        "date": _parse_dates(raw[_pick(cols, DATE_KEYS)]),
        "close": raw[_pick(cols, CLOSE_KEYS)],
        "vix": raw[_pick(cols, VIX_KEYS)],
    })
    return validate(df)


def load_two_nse_files(index_csv: str | Path, vix_csv: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Merge an index-history CSV and an India VIX CSV (both with Date, Close)."""
    ia, va = pd.read_csv(index_csv), pd.read_csv(vix_csv)
    ci, cv = {_norm(c): c for c in ia.columns}, {_norm(c): c for c in va.columns}
    left = pd.DataFrame({"date": _parse_dates(ia[_pick(ci, DATE_KEYS)]), "close": ia[_pick(ci, ("close",))]})
    right = pd.DataFrame({"date": _parse_dates(va[_pick(cv, DATE_KEYS)]), "vix": va[_pick(cv, ("close",))]})
    return validate(left.merge(right, on="date", how="inner"))
