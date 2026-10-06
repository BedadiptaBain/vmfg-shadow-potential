#!/usr/bin/env python3
"""07 - Can the Section 10 statistics tell a reflected ex-ante premium from an
unbounded one?  (Synthetic markets; see src/vmfg/empirical/identification.py.)

For each hypothesis (unbounded OU premium / OU premium reflected at zero) we
simulate R samples of 1,964 sessions and compute, for the true premium and for
both observable proxies, the statistics printed in Section 10.  Reported:
5th percentile / median / 95th percentile across samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import diagnostics as G, identification as I  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
R = 400
fmt = lambda t, d=3: "/".join(f"{v:.{d}f}" if np.isfinite(v) else "-" for v in t)

lines = ["# 07 - Identification study", "",
         f"R = {R} synthetic samples of n = 1,964 sessions per hypothesis. Entries are 5th pct / median / 95th pct.",
         "Paper (Section 10): 17.5% non-positive, 53 episodes, minimum -0.535, "
         "p = 0.556 / 0.457 / 0.682 / 0.467 at x = 0.02 / 0.05 / 0.10 / 0.20.", ""]
summaries = {}
for hyp in ("unbounded", "reflected"):
    panel = I.simulate_panel(R=R, premium=hyp, seed=2026 if hyp == "unbounded" else 2027)
    lines += [f"## Ex-ante premium: {hyp}" + (" (never negative)" if hyp == "reflected" else ""), "",
              "| series | share <= 0 | minimum | episodes | p(0.02) | p(0.05) | p(0.10) | p(0.20) |",
              "|---|---|---|---|---|---|---|---|"]
    for series, label in (("pi", "true ex-ante premium"), ("lam_back", "proxy, trailing RV"),
                          ("lam_fwd", "proxy, forward RV")):
        S = I.summarise(panel, series)
        summaries[(hyp, series)] = S
        ps = " | ".join(fmt(S["p"][x], 2) for x in G.PAPER_THRESHOLDS)
        lines.append(f"| {label} | {fmt(S['frac_nonpos'])} | {fmt(S['min'])} | {fmt(S['episodes'], 0)} | {ps} |")
    lines.append("")

ub, rf = summaries[("unbounded", "lam_back")], summaries[("reflected", "lam_back")]
lines += ["## Reading", "",
          "- Under the reflected hypothesis the true premium is never negative, yet both proxies are non-positive on "
          f"{fmt(rf['frac_nonpos'])} of sessions, reach minima of {fmt(rf['min'])}, and show conditional "
          "probabilities of the same size as Table 4 whenever the thresholds are reached.",
          "- The unbounded and reflected hypotheses produce overlapping distributions for every Section 10 statistic. "
          "The statistics are driven by realised-variance forecast errors, not by the sign of the ex-ante premium.",
          "- A test of Theorem 5.3's support prediction needs an ex-ante premium estimate, IV2_t - E_t[RV], with a "
          "forecasting model (e.g. HAR-RV on NIFTY returns) and an allowance for its forecast error.",
          "- The calibration here is illustrative (log-variance AR(1) with volatility spikes, Student-t returns); "
          "the identification point does not depend on it.", ""]
(OUT / "07_identification.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panel_r = I.simulate_panel(R=1, premium="reflected", seed=5)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(panel_r["lam_back"][0], lw=0.7, label="observable proxy IV2 - trailing RV")
    ax.plot(panel_r["pi"][0], lw=1.2, label="true ex-ante premium (reflected at 0)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("session")
    ax.set_ylabel("annualised variance units")
    ax.set_title("A premium that never goes negative, seen through the Section 10 proxy")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "07_identification.png", dpi=150)
except Exception as exc:
    print("figure skipped:", exc)
