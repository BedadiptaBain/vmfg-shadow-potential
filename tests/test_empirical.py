"""Section 10 machinery on toy inputs."""
import numpy as np
import pandas as pd

from vmfg.empirical import data as D, diagnostics as G, premium as P


def test_runs_and_conditional_table():
    lam = np.array([0.1, -0.03, -0.07, 0.02, -0.01, -0.3, -0.5, 0.2])
    assert G.runs_nonpositive(lam) == [2, 3]          # runs at t = 1-2 and t = 4-6
    tab = G.conditional_table(lam, (0.02, 0.2)).set_index("x")
    # lam_t < -0.02 at t = 1, 2, 5, 6 (t = 6 has a successor); next < -0.04 at t = 1, 5
    assert tab.loc[0.02, "n"] == 4 and tab.loc[0.02, "k"] == 2
    assert tab.loc[0.2, "n"] == 2 and tab.loc[0.2, "k"] == 1


def test_backward_forward_alignment():
    r = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    rvb = P.realized_variance(r, window=2, ann=2.0, direction="backward")
    rvf = P.realized_variance(r, window=2, ann=2.0, direction="forward")
    assert rvb.iloc[2] == 1 + 4 and rvb.iloc[5] == 16 + 25          # r_{t-1}^2 + r_t^2
    assert rvf.iloc[1] == 4 + 9 and np.isnan(rvf.iloc[4])           # r_{t+1}^2 + r_{t+2}^2


def test_loader_fixes_order_duplicates_and_units(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=30)
    df = pd.DataFrame({"Date": dates.strftime("%d-%m-%Y"), "NIFTY": np.linspace(100, 130, 30),
                       "VIX": np.full(30, 0.15)})
    df = pd.concat([df, df.iloc[[3]]]).sample(frac=1, random_state=0)
    f = tmp_path / "daily.csv"
    df.to_csv(f, index=False)
    out, rep = D.load_daily(f)
    assert rep.n_duplicates == 1 and not rep.input_was_sorted and rep.vix_rescaled
    assert out["date"].is_monotonic_increasing and np.allclose(out["vix"], 15.0)


def test_printed_table4_consistency():
    t = G.table4_consistency()
    prev = t[t.draft == "previous draft"].set_index("x")
    assert prev.loc[0.02, "attainable_k"] == "none" and bool(prev.loc[0.02, "n_exceeds_nonpositive_sessions"])
    cur = t[t.draft == "current draft"].set_index("x")
    assert cur.loc[0.02, "attainable_k"] == [50]


def test_sample_size_inconsistent_with_2020_2026():
    lo, hi = G.session_count_range("2020-01-01", "2026-12-31")
    assert hi < 1964
