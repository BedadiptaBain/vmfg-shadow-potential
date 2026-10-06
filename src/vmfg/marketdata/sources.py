"""Download RAW market data (nothing is transformed here; see build.py).

Every file is written byte-for-byte as served to data/raw/<source>/... and a
provenance line {url, file, bytes, sha256, utc} is appended to
data/raw/provenance.jsonl, so anyone can check that they processed the same
bytes.

Sources
  github        third-party mirrors on GitHub pinned to exact commits
                (reproducible byte-for-byte; the only route that works in
                restricted sandboxes)
  yahoo         Yahoo Finance via yfinance: ^NSEI, ^INDIAVIX, SPY, ^GSPC, ^VIX
  nse           NSE India public JSON endpoints: NIFTY 50 and India VIX history
  niftyindices  niftyindices.com historical NIFTY 50 values
  kite          Zerodha Kite Connect historical candles (needs api_key + access_token)
  fred          FRED CSV: SP500, VIXCLS
  cboe          CBOE daily VIX history CSV

The NSE and niftyindices endpoints are undocumented and change from time to
time; the functions fail loudly (HTTP error or empty payload) rather than
silently writing bad files.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------------------------------------------------------------------------
# pinned GitHub mirrors (commit SHAs recorded on 17 Sept 2026)
# ---------------------------------------------------------------------------
GITHUB_PINNED: dict[str, dict] = {
    "nifty50_1min": {
        "repo": "technovusin/nifty50-historical-data",
        "commit": "20a658f0e0ea0c84e33903d32b39caa724765f9a",
        "files": [f"1min/{y}/NIFTY50_1min_{y}.csv" for y in range(2017, 2026)] + [
            "1min/2026/NIFTY50_1min_20260101_to_20260908.csv",
            "1min/2026/NIFTY50_1min_20260909.csv",
            "1min/2026/NIFTY50_1min_20260910.csv",
            "1min/2026/NIFTY50_1min_20260911.csv",
            "1min/2026/NIFTY50_1min_20260915.csv",
            "1min/2026/NIFTY50_1min_20260916.csv",
        ],
        "content": "NIFTY 50 one-minute OHLC bars, 2017-04-03 to 2026-09-16, IST timestamps",
    },
    "indiavix_yahoo": {
        "repo": "prakash-ukhalkar/india-vix-bank-nifty-regime",
        "commit": "ca6450a3159b93d25132edf7b766617be3a876bf",
        "files": ["data/raw/india_vix.csv"],
        "content": "India VIX daily close/high/low from Yahoo (^INDIAVIX), 2016-01-01 to 2026-03-30",
    },
    "nifty_vix_yahoo_2020": {
        "repo": "tanishkamalik/NIFTY-50-FORECASTING",
        "commit": "0651e5e2bea5e8e59364c29a204eeb5266450d84",
        "files": ["outputs/forecasts/test_volatility_forecasts.csv",
                  "outputs/forecasts/garch_vix_comparison.csv"],
        "content": "Yahoo ^NSEI official-close log returns (x100) and ^INDIAVIX closes, 2020-01-02 to 2026-08-26",
    },
    "vix_cboe": {
        "repo": "datasets/finance-vix",
        "commit": "f430b9d23554096571a6729fd069c2824617f7fe",
        "files": ["data/vix-daily.csv"],
        "content": "CBOE VIX daily OHLC, 1990-01-02 to 2026-09-15",
    },
    "spy_yahoo": {
        "repo": "HuseinHaji/volatility-regime-prediction",
        "commit": "06ae511bfd00d3ad5a36ae002219171566cf90a4",
        "files": ["data/SPY_Yahoo_yfinance.csv"],
        "content": "SPY daily OHLC and adjusted close from Yahoo, 2005-02-25 to 2026-06-30",
    },
}

YAHOO_TICKERS = {"india": {"close": "^NSEI", "vix": "^INDIAVIX"},
                 "us": {"close": "SPY", "index": "^GSPC", "vix": "^VIX"}}
KITE_TOKENS = {"NIFTY 50": 256265, "INDIA VIX": 264969}
FRED_SERIES = ("SP500", "VIXCLS")
CBOE_SYMBOLS = ("VIX",)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _log(raw_root: Path, rec: dict) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    with open(raw_root / "provenance.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _save(raw_root: Path, dest: Path, url: str, content: bytes, note: str = "") -> Path:
    if not content:
        raise RuntimeError(f"empty response from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    _log(raw_root, dict(url=url, file=str(dest.relative_to(raw_root)), bytes=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                        utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), note=note))
    return dest


def fetch(url: str, dest: Path, raw_root: Path, headers: dict | None = None, refresh: bool = False,
          retries: int = 3, timeout: int = 60, session=None, method: str = "GET",
          data: bytes | None = None, note: str = "") -> Path:
    """Download url to dest (skipped if dest exists and refresh is False)."""
    dest = Path(dest)
    if dest.exists() and not refresh:
        return dest
    err = None
    for attempt in range(retries):
        try:
            if session is not None:
                resp = session.request(method, url, headers=headers, data=data, timeout=timeout)
                resp.raise_for_status()
                content = resp.content
            else:
                req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})},
                                             data=data, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    content = r.read()
            return _save(raw_root, dest, url, content, note)
        except Exception as exc:  # noqa: BLE001 - report the last error
            err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {err}")


def _chunks(start: dt.date, end: dt.date, days: int):
    a = start
    while a <= end:
        b = min(end, a + dt.timedelta(days=days - 1))
        yield a, b
        a = b + dt.timedelta(days=1)


def _as_date(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x))


# ---------------------------------------------------------------------------
# source 1: pinned GitHub mirrors
# ---------------------------------------------------------------------------
def download_github(raw_root: Path, which=None, refresh: bool = False) -> list[Path]:
    out = []
    for key, spec in GITHUB_PINNED.items():
        if which and key not in which:
            continue
        for path in spec["files"]:
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{path}"
            dest = raw_root / "github" / key / path
            out.append(fetch(url, dest, raw_root, refresh=refresh,
                             note=f"{spec['repo']}@{spec['commit'][:10]}: {spec['content']}"))
    return out


# ---------------------------------------------------------------------------
# source 2: Yahoo Finance (pip install yfinance)
# ---------------------------------------------------------------------------
def download_yahoo(raw_root: Path, market: str, start, end, refresh: bool = False) -> list[Path]:
    import yfinance as yf  # optional dependency
    import pandas as pd

    out = []
    for role, tic in YAHOO_TICKERS[market].items():
        dest = raw_root / "yahoo" / market / f"{tic.replace('^', '')}.csv"
        if dest.exists() and not refresh:
            out.append(dest)
            continue
        df = yf.download(tic, start=str(start), end=str(_as_date(end) + dt.timedelta(days=1)),
                         auto_adjust=False, actions=True, progress=False)
        if df is None or len(df) == 0:
            raise RuntimeError(f"Yahoo returned no rows for {tic}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out.append(_save(raw_root, dest, f"yfinance:{tic}", df.to_csv().encode(), note=f"{role} {tic}"))
    return out


# ---------------------------------------------------------------------------
# source 3: NSE India (official, from an Indian IP works best)
# ---------------------------------------------------------------------------
NSE_HOME = "https://www.nseindia.com"
NSE_VIX = NSE_HOME + "/api/historical/vixhistory?from={f}&to={t}"
NSE_INDEX = NSE_HOME + "/api/historical/indicesHistory?indexType={name}&from={f}&to={t}"


def _nse_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "application/json, text/plain, */*",
                      "Accept-Language": "en-US,en;q=0.9",
                      "Referer": NSE_HOME + "/reports-indices-historical-index-data"})
    s.get(NSE_HOME, timeout=30)                                      # sets the cookies NSE requires
    s.get(NSE_HOME + "/reports-indices-historical-index-data", timeout=30)
    return s


def download_nse(raw_root: Path, start, end, chunk_days: int = 90, pause: float = 1.0,
                 refresh: bool = False) -> list[Path]:
    s = _nse_session()
    out = []
    for a, b in _chunks(_as_date(start), _as_date(end), chunk_days):
        f, t = a.strftime("%d-%m-%Y"), b.strftime("%d-%m-%Y")
        tag = f"{a:%Y%m%d}_{b:%Y%m%d}"
        out.append(fetch(NSE_VIX.format(f=f, t=t), raw_root / "nse" / f"vix_{tag}.json", raw_root,
                         session=s, refresh=refresh, note="India VIX history"))
        out.append(fetch(NSE_INDEX.format(name=urllib.parse.quote("NIFTY 50"), f=f, t=t),
                         raw_root / "nse" / f"nifty50_{tag}.json", raw_root,
                         session=s, refresh=refresh, note="NIFTY 50 history"))
        time.sleep(pause)
    return out


# ---------------------------------------------------------------------------
# source 4: niftyindices.com (NIFTY 50 closes only)
# ---------------------------------------------------------------------------
NI_PAGE = "https://www.niftyindices.com/reports/historical-data"
NI_URL = "https://www.niftyindices.com/Backpage.aspx/getHistoricaldatatabletoString"


def download_niftyindices(raw_root: Path, start, end, pause: float = 1.0, refresh: bool = False) -> list[Path]:
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Content-Type": "application/json; charset=UTF-8",
                      "Origin": "https://www.niftyindices.com", "Referer": NI_PAGE,
                      "X-Requested-With": "XMLHttpRequest"})
    s.get(NI_PAGE, timeout=30)
    out = []
    for a, b in _chunks(_as_date(start), _as_date(end), 365):
        cinfo = ("{'name':'NIFTY 50','startDate':'%s','endDate':'%s','indexName':'NIFTY 50'}"
                 % (a.strftime("%d-%b-%Y"), b.strftime("%d-%b-%Y")))
        body = json.dumps({"cinfo": cinfo}).encode()
        out.append(fetch(NI_URL, raw_root / "niftyindices" / f"nifty50_{a:%Y%m%d}_{b:%Y%m%d}.json",
                         raw_root, session=s, method="POST", data=body, refresh=refresh,
                         note="niftyindices NIFTY 50 history"))
        time.sleep(pause)
    return out


# ---------------------------------------------------------------------------
# source 5: Zerodha Kite Connect
# ---------------------------------------------------------------------------
def download_kite(raw_root: Path, api_key: str, access_token: str, start, end,
                  interval: str = "day", refresh: bool = False) -> list[Path]:
    """Day candles: up to ~2000 days per request; minute candles: 60 days."""
    headers = {"X-Kite-Version": "3", "Authorization": f"token {api_key}:{access_token}"}
    span = {"day": 1900, "minute": 55, "5minute": 95}[interval]
    out = []
    for name, token in KITE_TOKENS.items():
        for a, b in _chunks(_as_date(start), _as_date(end), span):
            q = urllib.parse.urlencode({"from": f"{a} 00:00:00", "to": f"{b} 23:59:59"})
            url = f"https://api.kite.trade/instruments/historical/{token}/{interval}?{q}"
            tag = name.replace(" ", "").lower()
            out.append(fetch(url, raw_root / "kite" / interval / f"{tag}_{a:%Y%m%d}_{b:%Y%m%d}.json",
                             raw_root, headers=headers, refresh=refresh, note=f"Kite {name} {interval}"))
            time.sleep(0.4)  # Kite allows 3 historical requests per second
    return out


# ---------------------------------------------------------------------------
# sources 6-7: FRED and CBOE (US)
# ---------------------------------------------------------------------------
def download_fred(raw_root: Path, refresh: bool = False) -> list[Path]:
    return [fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                  raw_root / "fred" / f"{sid}.csv", raw_root, refresh=refresh, note=f"FRED {sid}")
            for sid in FRED_SERIES]


def download_cboe(raw_root: Path, refresh: bool = False) -> list[Path]:
    return [fetch(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv",
                  raw_root / "cboe" / f"{sym}_History.csv", raw_root, refresh=refresh, note=f"CBOE {sym}")
            for sym in CBOE_SYMBOLS]
