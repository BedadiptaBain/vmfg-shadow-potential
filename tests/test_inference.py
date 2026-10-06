"""Inference methods: bootstrap calibration, ADF behaviour, gap handling."""
import numpy as np
import pandas as pd

from vmfg.empirical import diagnostics as G, inference as F, realdata as RD


def test_block_bootstrap_matches_iid_standard_error():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 1.0, 4000)
    out = F.bootstrap_series(x, (0.5,), L=5, B=400, seed=1)
    # s.e. of the mean of iid N(0.3,1) with n = 4000 is 1/sqrt(4000) = 0.0158
    assert abs(out["se"]["mean"] - 1 / np.sqrt(4000)) < 0.003


def test_adf_distinguishes_ar1_from_random_walk():
    rng = np.random.default_rng(1)
    e = rng.normal(size=3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    rw = np.cumsum(e)
    assert F.adf(ar, p=5)["reject_unit_root"]
    assert not F.adf(rw, p=5)["reject_unit_root"]


def test_gaps_break_runs_and_pairs():
    lam = pd.Series([0.1, -0.3, np.nan, -0.3, -0.5, 0.2], index=pd.bdate_range("2020-01-01", periods=6))
    brk, close = RD.stats(lam, gaps="break"), RD.stats(lam, gaps="close")
    assert brk["episodes"] == 2 and close["episodes"] == 1        # the gap splits the run
    tb = brk["table"].set_index("x")
    tc = close["table"].set_index("x")
    # h = 0.2, observed pairs only: t=3 (-0.3 -> -0.5, below -0.4) and t=4 (-0.5 -> 0.2)
    assert tb.loc[0.2, "n"] == 2 and tb.loc[0.2, "k"] == 1
    # the pre-audit convention also pairs t=1 (-0.3) with t=3 across the gap
    assert tc.loc[0.2, "n"] == 3 and tc.loc[0.2, "k"] == 1


def test_mc_pvalue_bounds():
    sim = np.arange(1, 2001, dtype=float)
    assert F.mc_pvalue_lower(sim, -5.0) == 1 / 2001
    assert abs(F.kappa_delta(0.96, 0.01, 252)[1] - 252 * 0.01 / 0.96) < 1e-12
