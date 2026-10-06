#!/usr/bin/env python3
"""08 - Validate Section 10 and the model on REAL data.

Prerequisites
  python scripts/00_download_data.py --source github        # builds data/processed/*_github.csv
  (optional) your own exports in data/user/: participant_oi.csv, chain_history.csv

Run
  python scripts/08_real_data_validation.py [--end 2026-08-26] [--n-paper 1964]

Writes
  results/08_real_data_validation.md       human-readable report
  results/08_*.csv                          every table as CSV
  results/08_india_premium.png              figure
  paper/table_*.tex, paper/section10_generated.tex   LaTeX generated from the same numbers
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.empirical import diagnostics as G, realdata as RD  # noqa: E402
from vmfg.marketdata import userfiles as U  # noqa: E402

RES, PAPER, PROC = ROOT / "results", ROOT / "paper", ROOT / "data" / "processed"
RES.mkdir(exist_ok=True)
PAPER.mkdir(exist_ok=True)

PAPER_VALUES = dict(n=1964, frac=0.175, episodes=53, longest=28, min=-0.535,
                    t4_n=[90, 46, 22, 15], t4_p=[0.556, 0.457, 0.682, 0.467])


def fmt_date(x) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def column(st: dict) -> dict:
    t = st["table"]
    return {
        "sessions n": st["n"],
        "first / last": f"{st['first']} / {st['last']}",
        "non-positive sessions": f"{st['n_nonpos']} ({100 * st['frac_nonpos']:.1f}%)",
        "episodes": st["episodes"],
        "longest episode": st["longest"],
        "minimum (date)": f"{st['min']:.4f} ({st['min_date']})",
        **{f"Table 4 x={x}: n / p": f"{n} / {p:.3f}" for x, n, p in zip(t["x"], t["n"], t["p"])},
    }


def paper_column() -> dict:
    pv = PAPER_VALUES
    return {"sessions n": pv["n"], "first / last": "2020 / 2026 (as stated)",
            "non-positive sessions": f"approx. {round(pv['frac'] * pv['n'])} (17.5%)", "episodes": pv["episodes"],
            "longest episode": pv["longest"], "minimum (date)": f"{pv['min']:.3f} (-)",
            **{f"Table 4 x={x}: n / p": f"{n} / {p:.3f}"
               for x, n, p in zip(G.PAPER_THRESHOLDS, pv["t4_n"], pv["t4_p"])}}


def match_flags(st: dict) -> dict:
    pv, t = PAPER_VALUES, st["table"]
    return dict(n=st["n"] == pv["n"], longest=st["longest"] == pv["longest"],
                episodes=st["episodes"] == pv["episodes"], min=round(st["min"], 3) == pv["min"],
                t4_n=[int(a) == b for a, b in zip(t["n"], pv["t4_n"])],
                t4_p=[round(float(a), 3) == b for a, b in zip(t["p"], pv["t4_p"])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-08-26", help="last date with India VIX in the pinned data")
    ap.add_argument("--n-paper", type=int, default=1964)
    ap.add_argument("--india", default=str(PROC / "india_daily_github.csv"))
    ap.add_argument("--us", default=str(PROC / "us_daily_github.csv"))
    ap.add_argument("--participant-oi", default=str(ROOT / "data/user/participant_oi.csv"))
    ap.add_argument("--chain", default=str(ROOT / "data/user/chain_history.csv"))
    a = ap.parse_args()
    if not Path(a.india).exists():
        sys.exit("processed data missing: run  python scripts/00_download_data.py --source github")
    END = pd.Timestamp(a.end)

    india = pd.read_csv(a.india, parse_dates=["date"]).set_index("date")
    d = RD.add_constructions(india)
    do = RD.add_constructions(india, use_official=True)
    audit = json.loads((PROC / "build_audit_india_github.json").read_text())

    # ------------------------------------------------------------------ reproduction
    def last_n(s):
        """Calendar slice spanning the last n valid values; unobserved sessions stay NaN."""
        v = s.loc[:END].dropna()
        start = v.index[-a.n_paper] if len(v) >= a.n_paper else v.index[0]
        return s.loc[start:v.index[-1]]
    samples = {
        "R1 last 1,964, trailing (paper's construction)": last_n(d["lam_back"]),
        "R2 last 1,964, forward window": last_n(d["lam_fwd"]),
        "R3 last 1,964, intraday RV": last_n(d["lam_intra"]),
        "R4 stated period 2020-, trailing": d["lam_back"].loc["2020-01-01":END],
        "R5 stated period, official closes": do["lam_back"].loc["2020-01-01":END],
        "R6 full sample 2017-, trailing": d["lam_back"].loc[:END],
    }
    stats = {k: RD.stats(v) for k, v in samples.items()}
    rep = pd.DataFrame({"Paper (Sec. 10)": paper_column(), **{k: column(s) for k, s in stats.items()}})
    rep.to_csv(RES / "08_reproduction.csv")
    flags = {k: match_flags(s) for k, s in stats.items()}
    r1 = stats["R1 last 1,964, trailing (paper's construction)"]
    r4 = stats["R4 stated period 2020-, trailing"]
    r5 = stats["R5 stated period, official closes"]
    r1_off = RD.stats(last_n(do["lam_back"]))

    # ------------------------------------------------------------------ model vs data
    mvd_rows, bench_r1 = [], None
    us = pd.read_csv(a.us, parse_dates=["date"]).set_index("date") if Path(a.us).exists() else None
    du = RD.add_constructions(us) if us is not None else None
    mvd_sets = {"India, R1": samples["R1 last 1,964, trailing (paper's construction)"],
                "India, R4": samples["R4 stated period 2020-, trailing"]}
    if du is not None:
        mvd_sets["US SPY/VIX 2005-2026"] = du["lam_back"]
        mvd_sets["US SPY/VIX 2020-2026"] = du["lam_back"].loc["2020-01-01":]
    mvd_stats = {}
    for name, lam in mvd_sets.items():
        st = RD.stats(lam)
        mvd_stats[name] = st
        reg = G.drift_regression(lam, lag=25)
        bench = RD.fitted_ou_benchmark(st, R=2000, seed=1)
        if name == "India, R1":
            bench_r1 = bench
        for pred, model_says, data_say, verdict in RD.verdicts(st, reg, RD.mean_hac(lam), bench):
            mvd_rows.append(dict(sample=name, prediction=pred, model=model_says, data=data_say, verdict=verdict))
    mvd = pd.DataFrame(mvd_rows)
    mvd.to_csv(RES / "08_model_vs_data.csv", index=False)

    # ------------------------------------------------------------------ ex-ante premium
    def exante_table(dd: pd.DataFrame, start: str, label: str) -> pd.DataFrame:
        rows = []
        y = dd["RVf"]
        for v in ("levels", "logs", "logs_median", "levels_iv"):
            F = RD.har_exante(dd, v, return_forecast=True)
            e = (y - F).loc[start:].dropna()
            pi = (dd["IV2"] - F).loc[e.index]
            runs = G.runs_nonpositive(pi.to_numpy())
            rows.append(dict(market=label, forecast=v, n=len(e),
                             oos_r2=1 - np.mean(e ** 2) / np.var(y.loc[e.index]),
                             median_error=e.median(), share_rv_below_forecast=np.mean(e < 0),
                             share_pi_nonpos=np.mean(pi <= 0), min_pi=pi.min(),
                             longest_run=max(runs) if runs else 0))
        return pd.DataFrame(rows)

    ex = exante_table(d, "2020-01-01", "India")
    if du is not None:
        ex = pd.concat([ex, exante_table(du, "2008-01-01", "US")], ignore_index=True)
    ex.to_csv(RES / "08_exante.csv", index=False)

    # ------------------------------------------------------------------ mechanism (user files)
    mech, speeds, uaudit = None, None, {}
    if Path(a.participant_oi).exists() and Path(a.chain).exists():
        poi, ua1 = U.load_participant_oi(Path(a.participant_oi))
        ch, ua2 = U.load_chain_history(Path(a.chain))
        uaudit = dict(participant_oi=ua1, chain_history=ua2)
        spot_gap = (ch["spot"] / india["close"].reindex(ch.index) - 1).abs()
        uaudit["chain_spot_vs_nifty_close_median_abs_gap"] = float(spot_gap.median())
        m = d[["IV2", "lam_back"]].join(
            poi[["log_short_detr", "seller_net_share", "fii_net_share"]], how="inner").join(
            ch[["log_vega_detr"]], how="left")
        wk = m.resample("W-FRI").mean().dropna()
        proxies = ["log_short_detr", "seller_net_share", "fii_net_share", "log_vega_detr"]
        mech = RD.mechanism_table(wk, ["IV2", "lam_back"], proxies, lag=8)
        mech.to_csv(RES / "08_mechanism.csv", index=False)
        speeds = pd.DataFrame([dict(series=c, weekly_phi=RD.ar1_phi(wk[c]),
                                    kappa_per_year=-52 * np.log(RD.ar1_phi(wk[c])))
                               for c in ["lam_back", *proxies]])
        speeds.to_csv(RES / "08_speeds.csv", index=False)

    # ------------------------------------------------------------------ figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        fig, ax = plt.subplots(2, 1, figsize=(10, 6.4))
        full = d["lam_back"].loc[:END]
        ax[0].plot(full.index, full.values, lw=0.6, color="C0", label="NIFTY: (VIX/100)$^2$ - trailing 21-session RV")
        s1 = samples["R1 last 1,964, trailing (paper's construction)"]
        ax[0].axvspan(s1.index[0], s1.index[-1], color="C1", alpha=0.10, label="last 1,964 sessions (reproduces the printed table)")
        ax[0].axvline(pd.Timestamp("2020-01-01"), color="k", ls=":", lw=0.8, label="1 January 2020")
        ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_ylabel("annualised variance")
        ax[0].legend(fontsize=7, loc="lower right")
        x = s1.to_numpy()
        bins = np.linspace(x.min(), x.max(), 120)
        ax[1].hist(x, bins=bins, density=True, alpha=0.6, label="empirical (R1)")
        g = np.linspace(x.min(), x.max(), 400)
        ax[1].plot(g, norm.pdf(g, x.mean(), x.std()), "k-", lw=1, label="Gaussian stationary law (same mean and s.d.)")
        ax[1].set_yscale("log")
        ax[1].set_ylim(1e-3, None)
        ax[1].set_xlabel("premium proxy")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(RES / "08_india_premium.png", dpi=150)
    except Exception as exc:
        print("figure skipped:", exc)

    # ------------------------------------------------------------------ LaTeX tables
    def tex_escape(s: str) -> str:
        return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
                .replace("_", "\\_").replace("#", "\\#").replace("~", "\\textasciitilde{}")
                .replace("^", "\\^{}").replace("<=", "$\\le$").replace("<", "$<$").replace(">", "$>$"))

    cols = ["Paper (Sec. 10)", "R1 last 1,964, trailing (paper's construction)", "R4 stated period 2020-, trailing",
            "R5 stated period, official closes", "R2 last 1,964, forward window"]
    heads = ["Paper", "R1", "R4", "R5", "R2"]
    body = "\n".join(" & ".join([tex_escape(i)] + [tex_escape(rep.loc[i, c]) for c in cols]) + r" \\"
                     for i in rep.index)
    (PAPER / "table_reproduction.tex").write_text(
        "\\begin{table}[t]\\centering\\scriptsize\n\\caption{Section~10 statistics: printed values and "
        "recomputation from the replication data. R1: last 1,964 sessions to " + a.end + ", trailing window "
        "(the construction behind the printed table). R4: 2020-01-01 onward. R5: R4 with official closes. "
        "R2: R1 with the forward window.}\\label{tab:repro}\n"
        "% requires \\usepackage{graphicx,booktabs}\n\\resizebox{\\linewidth}{!}{%\n"
        "\\begin{tabular}{l" + "l" * len(cols) + "}\\toprule\n & " + " & ".join(heads) + r" \\ \midrule" + "\n"
        + body + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    mv = mvd[mvd["sample"] == "India, R1"]
    (PAPER / "table_model_vs_data.tex").write_text(
        "\\begin{table}[t]\\centering\\scriptsize\n\\caption{Predictions of Theorem~5.3 and Section~8 against the "
        "NIFTY premium proxy (sample R1).}\\label{tab:mvd}\n\\begin{tabular}{p{3.3cm}p{3.2cm}p{4.6cm}p{1.9cm}}\\toprule\n"
        "prediction & model & data & verdict \\\\ \\midrule\n"
        + "\n".join(f"{tex_escape(r.prediction)} & {tex_escape(r.model)} & {tex_escape(r.data)} & {tex_escape(r.verdict)} \\\\"
                    for r in mv.itertuples())
        + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (PAPER / "table_exante.tex").write_text(
        "\\begin{table}[t]\\centering\\scriptsize\n\\caption{Out-of-sample HAR forecasts of the forward "
        "21-session realised variance and the implied ex-ante premium $\\hat\\pi_t=\\mathrm{IV}^2_t-F_t$.}"
        "\\label{tab:exante}\n\\begin{tabular}{llrrrrrr}\\toprule\n"
        "market & forecast & $R^2_{oos}$ & median error & share $RV<F$ & share $\\hat\\pi\\le0$ & min $\\hat\\pi$ & longest run \\\\ \\midrule\n"
        + "\n".join(f"{r.market} & {tex_escape(r.forecast)} & {r.oos_r2:.3f} & {r.median_error:+.4f} & "
                    f"{r.share_rv_below_forecast:.2f} & {r.share_pi_nonpos:.2f} & {r.min_pi:+.3f} & {r.longest_run} \\\\"
                    for r in ex.itertuples())
        + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    if mech is not None:
        (PAPER / "table_mechanism.tex").write_text(
            "\\begin{table}[t]\\centering\\scriptsize\n\\caption{Weekly co-movement of the premium with "
            "deployment proxies (Newey--West $t$, 8 lags). Assumption~2.1 with a constant intercept implies "
            "correlation $-1$.}\\label{tab:mech}\n\\begin{tabular}{llrrrr}\\toprule\n"
            "premium & proxy & corr (levels) & $t$ & corr (changes) & $t$ \\\\ \\midrule\n"
            + "\n".join(f"{tex_escape(r.premium)} & {tex_escape(r.proxy)} & {r.corr_levels:+.2f} & {r.t_levels:+.2f} & "
                        f"{r.corr_changes:+.2f} & {r.t_changes:+.2f} \\\\" for r in mech.itertuples())
            + "\n\\bottomrule\\end{tabular}\\end{table}\n")

    # ------------------------------------------------------------------ generated Section 10
    exI = ex[ex.market == "India"].set_index("forecast")

    def sci(x: float) -> str:
        mant, expo = f"{x:.0e}".split("e")
        return f"${mant}\\times10^{{{int(expo)}}}$"

    min_sentence = ("The observed minimum lies below all 2{,}000 simulated minima of the fitted Gaussian OU."
                    if bench_r1["min_rank"] == 0 else
                    f"The observed minimum lies at the {100 * bench_r1['min_rank']:.1f}th percentile of the "
                    "minima of the fitted Gaussian OU.")
    f1 = flags["R1 last 1,964, trailing (paper's construction)"]
    sec = rf"""% ---- generated by scripts/08_real_data_validation.py; do not edit numbers by hand ----
\section{{Empirical illustration}}\label{{sec:empirics}}

\paragraph{{Data.}} One-minute NIFTY~50 bars from {audit['first_date']} to {audit['last_date']}
({audit['minute_bars']:,} bars) and daily India VIX closes from Yahoo Finance; two independent
downloads of the latter agree to {sci(audit['max_abs_diff_on_overlap'])} on {audit['overlap']:,} common
days. Yahoo official-close NIFTY returns (2020--2026) serve as a cross-check. Every file is fetched,
hashed and logged by \texttt{{scripts/00\_download\_data.py}} from pinned sources; the processing below
is \texttt{{src/vmfg/marketdata/build.py}}.

\paragraph{{Processing.}} (i) Bars outside 09:15--15:30 IST are dropped. (ii) A session is a weekday
with at least 120 such bars, or one of {len(audit['special_sessions_kept'])} documented weekend live
sessions; this removes {len(audit['dropped_days'])} mock-trading and muhurat days. (iii) The daily close
is the mean of the one-minute closes from 15:00 to 15:29, mirroring NSE's closing-price rule; its log
returns track official returns with correlation {audit['official_check_corr']:.4f} and RMS gap
{audit['official_check_rms_bp']:.1f}\,bp over {audit['official_check_n']:,} sessions. (iv)
$r_t=\log(C_t/C_{{t-1}})$ on this calendar. (v) India VIX is attached without filling;
{audit['sessions_missing_vix']} sessions lack it and are excluded.

\paragraph{{Construction.}} With $\mathrm{{IV}}^2_t=(\mathrm{{VIX}}_t/100)^2$ the premium proxy is
\[
\hat\lambda_t=\mathrm{{IV}}^2_t-\frac{{252}}{{21}}\sum_{{j=0}}^{{20}}r_{{t-j}}^2 ,
\]
a trailing window known at the close of day $t$. A non-positive episode is a maximal run of consecutive
sessions with $\hat\lambda_t\le0$, and for a threshold $h$ Table~\ref{{tab:repro}} reports
$n_h=\#\{{t:\hat\lambda_t<-h\}}$ and $p_h=\Pr(\hat\lambda_{{t+1}}<-2h\mid\hat\lambda_t<-h)$.

\paragraph{{Results.}} The last {r1['n']:,} sessions, {r1['first']} to {r1['last']} (sample R1), give
{r1['n_nonpos']} non-positive sessions ({100 * r1['frac_nonpos']:.1f}\%) in {r1['episodes']} episodes,
the longest lasting {r1['longest']} sessions, and a minimum of ${r1['min']:.3f}$ on {r1['min_date']}
(${r1_off['min']:.4f}$ with official closes). The counts $n_h$ are
{', '.join(str(int(x)) for x in r1['table']['n'])} and the probabilities $p_h$ are
{', '.join(f"{x:.3f}" for x in r1['table']['p'])}. Restricting to 2020 onward (R4) leaves {r4['n']:,}
sessions, of which {100 * r4['frac_nonpos']:.1f}\% are non-positive, in {r4['episodes']} episodes.

\paragraph{{What the data say about the model.}} Table~\ref{{tab:mvd}} confronts the proxy with the
predictions of Theorem~5.3. The proxy crosses zero repeatedly, as unbounded support requires. Its law,
however, is far from the Gaussian stationary law: skewness ${r1['skew']:+.1f}$, excess kurtosis
{r1['exkurt']:.0f}. A Gaussian with the same mean and standard deviation would place
{100 * r1['gauss_frac_nonpos']:.0f}\% of its mass at or below zero, against
{100 * r1['frac_nonpos']:.1f}\% observed. {min_sentence} The negative
skewness has the sign predicted by concave clearing (Corollary~8.4).

The proxy is not the ex-ante premium. Out-of-sample HAR forecasts give ex-ante premia that are
non-positive on {100 * exI.loc['logs_median', 'share_pi_nonpos']:.0f}\% to
{100 * exI.loc['levels', 'share_pi_nonpos']:.0f}\% of sessions, depending on the forecast. Every
forecast has a median error (${exI['median_error'].min():+.4f}$ to ${exI['median_error'].max():+.4f}$) of
the same order as the premium itself (Table~\ref{{tab:exante}}). The sign of the ex-ante premium on a
given day is therefore not identified by these data, and neither is reflection at zero.

Finally, Assumption~2.1 with a constant intercept makes $\lambda$ an exact decreasing function of
deployment, so the two should be perfectly negatively correlated. Weekly correlations with four
deployment proxies built from NSE participant-wise open interest and the option chain are weak and
change sign across proxies (Table~\ref{{tab:mech}}). This points to demand shocks, which the model
omits; they can be added as a second linear state without losing the reduction.
"""
    (PAPER / "section10_generated.tex").write_text(sec)

    # ------------------------------------------------------------------ markdown report
    L = ["# 08 - Section 10 and the model on real data", "",
         f"Data: NIFTY 50 one-minute bars {audit['first_date']} to {audit['last_date']}, India VIX (Yahoo), "
         f"official-close cross-check, US SPY/CBOE VIX, and the author's participant-OI and option-chain exports. "
         f"End date for India VIX: {a.end}.", "",
         "## Data audit", "", "```", json.dumps({k: v for k, v in audit.items() if k != 'missing_vix_dates'}, indent=1),
         json.dumps(uaudit, indent=1), "```", "",
         "## Reproduction of the printed Section 10", "", rep.to_markdown(), "",
         "### Does the recomputation match the printed numbers? (sample R1)", "",
         f"- n = 1,964: {f1['n']}; longest episode 28: {f1['longest']}; episodes 53: {f1['episodes']} "
         f"(got {r1['episodes']}); minimum -0.535: {f1['min']} (got {r1['min']:.4f}; official closes give {r1_off['min']:.4f})",
         f"- Table 4 n column (90, 46, 22, 15): {f1['t4_n']}",
         f"- Table 4 p column (0.556, 0.457, 0.682, 0.467): {f1['t4_p']}",
         f"- share non-positive: {100 * r1['frac_nonpos']:.1f}% vs 17.5%", "",
         "## Model vs data", "", mvd.to_markdown(index=False), "",
         "Fitted-OU benchmark for R1 (5th/50th/95th percentiles over 2000 simulated samples): "
         f"share <= 0 {bench_r1['frac_band']}, minimum {bench_r1['min_band']}, "
         f"Table 4 p bands {{x: band}} = { {k: v['p_band'] for k, v in bench_r1['table'].items()} }", "",
         "## Ex-ante premium (HAR forecasts, out of sample)", "", ex.round(4).to_markdown(index=False), ""]
    if mech is not None:
        L += ["## Mechanism: premium versus deployment (weekly, Newey-West t)", "", mech.round(3).to_markdown(index=False), "",
              "AR(1) speeds (weekly): the model gives lam and Y the same kappa.", "", speeds.round(3).to_markdown(index=False), ""]
    (RES / "08_real_data_validation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
