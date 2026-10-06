#!/usr/bin/env python3
"""06 - Section 10: what can be checked without data, and a full recomputation with data.

Usage
  python scripts/06_empirical_section10.py                      # checks on the printed numbers only
  python scripts/06_empirical_section10.py --data DAILY.csv     # + full recomputation
  python scripts/06_empirical_section10.py --data Manifold_Research_Data.xlsx --sheet DAILY
  python scripts/06_empirical_section10.py --index nifty.csv --vix indiavix.csv
Options: --start / --end restrict the sample (e.g. --start 2020-01-01 to match the text),
         --window (default 21), --ann (default 252).
Outputs: results/06_section10_printed_numbers.md and, with data, results/06_section10_data.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import data as D, diagnostics as G, premium as P  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def printed_number_checks() -> str:
    L = ["# 06 - Section 10: checks on the printed numbers (no data needed)", ""]
    # 1. Table 4 lattice and subset constraints
    t = G.table4_consistency()
    L += ["## Table 4 internal consistency", "",
          "`attainable_k` lists integers k with round(k/n, 3) = p; `n_exceeds_nonpositive_sessions` flags n larger than "
          f"the {int(np.floor((G.PAPER_FRAC_NONPOS + 0.0005) * G.PAPER_N))} non-positive sessions implied by "
          "17.5% of 1,964 (sessions with lam < -x are a subset of those with lam <= 0).", "",
          "| draft | x | n | p | attainable k | n > #non-positive sessions |", "|---|---|---|---|---|---|"]
    for r in t.itertuples():
        L.append(f"| {r.draft} | {r.x} | {r.n} | {r.p} | {r.attainable_k} | {r.n_exceeds_nonpositive_sessions} |")
    L += ["", "- The two drafts print **identical probabilities with different n**. The previous n column is impossible "
          "(n = 389 at x = 0.02 exceeds the ~344 non-positive sessions, and 0.556 is not k/389 for any integer k). "
          "The current n column is arithmetically possible but, per the draft's own note, was never recomputed. "
          "**Table 4 must be regenerated from data.**", ""]
    # 2. session counts
    L += ["## Sample size versus the stated period", ""]
    for start, end in (("2020-01-01", "2026-08-31"), ("2020-01-01", "2026-12-31")):
        lo, hi = G.session_count_range(start, end)
        L.append(f"- NSE sessions {start} to {end}: about {lo}-{hi} "
                 f"(weekdays {G.weekday_count(start, end)} minus 11-17 weekday holidays a year).")
    need = G.PAPER_N + 21
    starts = pd.date_range("2017-01-01", "2020-01-01", freq="MS")
    fits = [s.strftime("%Y-%m") for s in starts
            if G.session_count_range(s.strftime("%Y-%m-%d"), "2026-09-11")[0] <= need
            <= G.session_count_range(s.strftime("%Y-%m-%d"), "2026-09-11")[1]]
    L += [f"- n = 1,964 usable sessions (+21 for the rolling window) ending mid-September 2026 needs a start around "
          f"**{fits[0]} to {fits[-1]}**, i.e. roughly eight years of history, not 2020-2026."
          if fits else "- no start date in 2017-2019 reproduces n = 1,964",
          f"- 17.5% of 1,964 = {0.175 * 1964:.1f}, so 343 or 344 non-positive sessions (both round to 17.5%).", ""]
    # 3. Gaussian consistency
    L += ["## Is the printed sample compatible with the Gaussian law of Theorem 5.3?", "",
          "If lam were Gaussian with mean m and P(lam <= 0) = 17.5%, its s.d. is m / Phi^-1(0.825) = m / 0.9346; "
          "the z-score of the printed minimum -0.535 is then:", "",
          "| assumed mean premium m | implied s.d. | z-score of the minimum |", "|---|---|---|"]
    for m in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
        z = G.gaussian_min_zscore(m)
        L.append(f"| {m} | {m / 0.9346:.4f} | {z:.1f} |")
    L += ["", "- For mean premia between 0.002 and 0.05 the minimum sits roughly 11 to 250 standard deviations out: the proxy's law is "
          "heavily left-tailed, which the Gaussian stationary law cannot produce. Only the *support* prediction is "
          "consistent with the data; the *shape* prediction is not (under the paper's own identification of the proxy "
          "with lam). With the corrected Section 8 sign, concave clearing at least pushes in the observed direction.", "",
          "## What Table 4 can and cannot show", "",
          "- For a hard floor F, Pr(lam_{t+1} < -2x | lam_t < -x) = 0 as soon as -2x < F. Every floor above the sample "
          "minimum is already excluded by the minimum itself; Table 4 adds no information about hard floors.",
          "- 17.5% non-positive sessions exclude reflection of the *proxy* at zero, but the model's lam is an ex-ante "
          "premium. Proxy = ex-ante premium minus a realised-variance forecast error (plus a lag error for the "
          "backward window). See 07_identification.md: a premium that is never negative ex ante reproduces all of "
          "Section 10's statistics.",
          "- The construction must state whether realised variance is trailing (the dashboard's Lambda_ante) or "
          "forward (the dashboard's Lambda_post), and that IV is India VIX (30 calendar days) against a "
          "21-session window.", ""]
    return "\n".join(L)


def data_report(df: pd.DataFrame, rep: D.ValidationReport, args) -> str:
    frame = P.build_premia(df, window=args.window, ann=args.ann).set_index("date")
    if args.start:
        frame = frame.loc[args.start:]
    if args.end:
        frame = frame.loc[:args.end]
    L = ["# 06 - Section 10 recomputed from data", "", "## Data validation", "", rep.as_markdown(), "",
         f"Window w = {args.window} sessions, annualisation m = {args.ann}. Sample used: "
         f"{frame.index[0].date()} to {frame.index[-1].date()}.", ""]
    for col, label in (("lam_back", "trailing RV (IV2_t - RV_{t-20..t})"), ("lam_fwd", "forward RV (IV2_t - RV_{t+1..t+21})")):
        st = G.section10_stats(frame[col])
        bs = G.block_bootstrap(frame[col].dropna().to_numpy(), L=args.window, B=args.boot)
        L += [f"## Proxy: {label}", "",
              f"- n = {st['n']} sessions; non-positive: {st['n_nonpos']} ({100 * st['frac_nonpos']:.1f}%, "
              f"block-bootstrap 90% interval {bs['frac_nonpos_90']}); episodes {st['episodes']}; longest {st['longest']}",
              f"- min {st['min']:.4f}; mean {st['mean']:.5f}; s.d. {st['sd']:.5f}; skewness {st['skew']:.2f}; "
              f"excess kurtosis {st['exkurt']:.1f}", "",
              "| x | n | k | p | 90% block-bootstrap interval | share of resamples where p is defined |",
              "|---|---|---|---|---|---|"]
        for r in st["table"].itertuples():
            L.append(f"| {r.x} | {r.n} | {r.k} | {r.p:.3f} | {bs['p_90'][r.x]} | {1 - bs['p_undefined_share'][r.x]:.2f} |")
        reg = G.drift_regression(frame[col], lag=max(25, args.window + 4))
        L += ["", "Drift regression d lam_{t+1} on [1, lam, lam^2, min(lam,0)] (Newey-West s.e.):", "",
              "| term | coef | s.e. | t |", "|---|---|---|---|"]
        L += [f"| {i} | {r.coef:.4g} | {r.nw_se:.3g} | {r.t:.2f} |" for i, r in reg.iterrows()]
        L.append("")
    per_year = frame.groupby(frame.index.year).size()
    L += ["## Sessions per calendar year", "", "| year | sessions |", "|---|---|"]
    L += [f"| {y} | {n} |" for y, n in per_year.items()]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--sheet", default="DAILY")
    ap.add_argument("--index")
    ap.add_argument("--vix")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--ann", type=float, default=252.0)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    text = printed_number_checks()
    (OUT / "06_section10_printed_numbers.md").write_text(text + "\n")
    print(text)

    if args.data or (args.index and args.vix):
        if args.data:
            df, rep = D.load_daily(args.data, sheet=args.sheet)
        else:
            df, rep = D.load_two_nse_files(args.index, args.vix)
        out = data_report(df, rep, args)
        (OUT / "06_section10_data.md").write_text(out)
        print(out)
    else:
        print("\nNo data supplied: pass --data <DAILY export> to recompute Section 10 (see data/README.md).")


if __name__ == "__main__":
    main()
