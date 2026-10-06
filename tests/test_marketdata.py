"""Processing rules R1-R7 and the ex-ante forecast, on synthetic inputs (no network)."""
import numpy as np
import pandas as pd

from vmfg.empirical import realdata as RD
from vmfg.marketdata import build as Bd, sources as Sr, userfiles as U


def _day(date, start="09:15", end="15:29", level=100.0, step=0.0):
    ts = pd.date_range(f"{date} {start}", f"{date} {end}", freq="1min")
    close = level + step * np.arange(len(ts))
    return pd.DataFrame({"Timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"), "Open": close, "High": close,
                         "Low": close, "Close": close})


def _bars(frames):
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["Timestamp"])
    df["date"] = df["ts"].dt.normalize()
    df["hm"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df


def test_sessions_close_and_returns():
    bars = _bars([
        _day("2022-01-06", level=100.0),                      # Thursday, full
        _day("2022-01-07", level=110.0, step=0.01),           # Friday, full, rising
        _day("2022-01-08", level=500.0),                      # Saturday mock session -> dropped
        _day("2022-01-10", start="18:15", end="19:15"),       # evening muhurat-like -> dropped
        _day("2022-01-11", start="13:45", end="14:44"),       # 60-bar weekday session -> dropped
        _day("2022-01-12", level=121.0),                      # Wednesday, full
    ])
    daily, audit = Bd.daily_from_minutes(bars)
    assert [d.strftime("%m-%d") for d in daily.index] == ["01-06", "01-07", "01-12"]
    # R3: mean of closes 15:00-15:29 (minutes 345..374 of the session)
    expected = 110.0 + 0.01 * np.arange(345, 375).mean()
    assert abs(daily.loc["2022-01-07", "close"] - expected) < 1e-9
    # R4: the return on 01-12 spans the removed days
    assert abs(daily.loc["2022-01-12", "r"] - np.log(121.0 / expected)) < 1e-12
    assert "2022-01-08" in audit["dropped_days"]


def test_rv5_on_known_path():
    bars = _bars([_day("2022-01-06", level=100.0, step=0.05)])
    daily, _ = Bd.daily_from_minutes(bars)
    closes = 100.0 + 0.05 * np.arange(375)
    bucket_close = closes[4::5]
    path = np.concatenate([[100.0], bucket_close])        # open, then 75 bucket closes
    assert abs(daily["rv5"].iloc[0] - np.sum(np.diff(np.log(path)) ** 2)) < 1e-15


def test_pinned_registry_is_complete():
    for key, spec in Sr.GITHUB_PINNED.items():
        assert len(spec["commit"]) == 40 and spec["files"], key


def test_userfile_loaders(tmp_path):
    poi = pd.DataFrame({"date": ["2020-01-02", "2020-01-01"], "client_long": [10, 12], "client_short": [8, 9],
                        "dii_long": [0, 0], "dii_short": [1, 1], "fii_long": [5, 4], "fii_short": [3, 3],
                        "pro_long": [2, 2], "pro_short": [5, 5], "net_short_pro": [3, 3],
                        "net_short_fii": [-2, -1], "net_short_client": [-2, -3]})
    poi.to_csv(tmp_path / "p.csv", index=False)
    df, audit = U.load_participant_oi(tmp_path / "p.csv")
    assert not audit["was_sorted"] and df.index.is_monotonic_increasing
    assert audit["long_equals_short_share"] == 1.0
    assert np.allclose(df["short_total"], [18, 17])      # rows re-sorted by date


def test_har_has_no_look_ahead():
    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                       "vix": 15 + rng.normal(0, 1, n)}, index=pd.bdate_range("2015-01-01", periods=n))
    df["r"] = np.log(df["close"]).diff()
    d = RD.add_constructions(df)
    F1 = RD.har_exante(d, "levels", return_forecast=True)
    d2 = d.copy()
    d2.iloc[800:, d2.columns.get_loc("RVf")] *= 50.0       # change only far-future targets
    F2 = RD.har_exante(d2, "levels", return_forecast=True)
    # forecasts at t < 800 + 21 cannot use targets s >= 800
    assert np.allclose(F1.iloc[:821].dropna(), F2.iloc[:821].dropna())
