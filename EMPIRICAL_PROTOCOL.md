# Empirical protocol: how every empirical number is produced

> Section, theorem and table numbers in this document refer to an earlier draft of the paper. The README maps every script to the final manuscript, whose empirical section is Section 13.

This document is the strategy and the recipe. Each step names the command, the code that implements it, and the file it writes. All data are fetched from pinned sources, so a verifier gets the same bytes and hence the same numbers.

## 0. What the section tests, and what it cannot

The model makes predictions about the equilibrium premium $\lambda_t$ and the aggregate deployment $Y_t$ (Theorems 6.3–6.4 and Proposition 14.2 of the final paper). The empirical section confronts them with data in three layers.

| layer | object | tests | why it is limited |
|---|---|---|---|
| A. printed statistics | proxy $\hat\lambda_t=\mathrm{IV}^2_t-RV_t$ | share of non-positive sessions, episodes, minimum, Table 4 | the proxy is not the model's ex-ante premium |
| B. model vs data | same proxy | P1 OU persistence and linear drift; P2 Gaussian law; P4 sign of skewness; P5 positive mean; P6 fitted-OU benchmark | the rolling window makes the proxy mechanically persistent |
| C1. ex-ante premium | $\hat\pi_t=\mathrm{IV}^2_t-F_t$ (HAR forecast) | P3 unbounded support | the sign depends on the forecast model |
| C2. mechanism | premium vs deployment proxies | M1 negative co-movement ($\lambda=a-bY$); M2 equal persistence of $\lambda$ and $Y$ | supply and demand are simultaneous |

## 1. Data

| id | content | source (pinned) | role |
|---|---|---|---|
| D1 | NIFTY 50 one-minute OHLC, 2017-04-03 → 2026-09-16, IST | GitHub `technovusin/nifty50-historical-data` @ `20a658f` | closes, returns, intraday RV |
| D2 | India VIX daily close, 2016-01-01 → 2026-03-30 | GitHub `prakash-ukhalkar/india-vix-bank-nifty-regime` @ `ca6450a` (Yahoo `^INDIAVIX`) | implied variance |
| D3 | India VIX daily close and official-close NIFTY log returns, 2020-01-02 → 2026-08-26 | GitHub `tanishkamalik/NIFTY-50-FORECASTING` @ `0651e5e` (Yahoo `^INDIAVIX`, `^NSEI`) | VIX extension; cross-check of D1 closes |
| D4 | CBOE VIX daily, 1990 → 2026-09-15 | GitHub `datasets/finance-vix` @ `f430b9d` | US comparison |
| D5 | SPY daily incl. adjusted close, 2005-02-25 → 2026-06-30 | GitHub `HuseinHaji/volatility-regime-prediction` @ `06ae511` (Yahoo) | US comparison |
| U1 | NSE participant-wise index-option OI, 2020-01-01 → 2026-08-14 | author's pipeline (`data/user/participant_oi.csv`) | deployment proxies |
| U2 | NIFTY option-chain daily summary, 2020-01-01 → 2026-08-14 | author's pipeline (`data/user/chain_history.csv`) | vega outstanding, ATM IV |

Primary sources can replace D1–D5 with the same code path: NSE, niftyindices.com, Yahoo (`yfinance`), Zerodha Kite (tokens 256265 = NIFTY 50, 264969 = INDIA VIX), FRED and CBOE. See `scripts/00_download_data.py --help`.

## 2. Step 0: download (raw bytes only)

```bash
python scripts/00_download_data.py --source github
```

- Code: `src/vmfg/marketdata/sources.py`.
- Every file goes to `data/raw/<source>/...` unchanged.
- One line per file is appended to `data/raw/provenance.jsonl`, recording the URL, bytes, SHA-256 and UTC time.
- A verifier compares these hashes with their own download.

## 3. Step 1: build the daily dataset

Code: `src/vmfg/marketdata/build.py`. Output: `data/processed/india_daily_github.csv` and `build_audit_india_github.json`.

| rule | action | reason |
|---|---|---|
| R1 | keep bars with IST time in [09:15, 15:30] | regular session only (drops pre-open and evening bars) |
| R2 | a session is a weekday with ≥ 120 such bars, or one of 6 documented weekend live sessions (budget days, 2024 disaster-recovery drills) | removes NSE mock-trading Saturdays and Diwali muhurat sessions (12 dates in D1) |
| R3 | daily close = mean of one-minute closes 15:00–15:29 | NSE's official close is a 15:00–15:30 average; this matches official returns with correlation 0.9987 and RMS gap 5.7 bp (the last bar gives 14.2 bp) |
| R4 | $r_t=\log(C_t/C_{t-1})$ on the R2 calendar | returns never span a removed session |
| R5 | $rv5_t$ = sum of squared 5-minute returns (from the 09:15 open); $r^{on}_t=\log(O_t/C_{t-1})$ | intraday variance |
| R6 | VIX from D2 ∪ D3; the two agree to 4e-15 on 1,528 common days | integrity check |
| R7 | VIX is left-joined onto the calendar with no filling; sessions without VIX (38) get no premium | no invented values |

The audit JSON lists:
- every dropped date and the special sessions kept;
- the missing-VIX dates;
- the five largest gaps against official returns. The largest is 24 Feb 2021, NSE's trading halt with an extended session.

## 4. Step 2: premium constructions

With $\mathrm{IV}^2_t=(\mathrm{VIX}_t/100)^2$, $m=252$, $w=21$:

$$\hat\lambda^{\rm back}_t=\mathrm{IV}^2_t-\frac{m}{w}\sum_{j=0}^{w-1}r_{t-j}^2 \quad\text{(trailing; used in the paper)}$$

$$\hat\lambda^{\rm fwd}_t=\mathrm{IV}^2_t-\frac{m}{w}\sum_{j=1}^{w}r_{t+j}^2,\qquad \hat\lambda^{\rm intra}_t=\mathrm{IV}^2_t-\frac{m}{w}\sum_{j=0}^{w-1}\big(rv5_{t-j}+(r^{on}_{t-j})^2\big).$$

Code: `realdata.add_constructions`.

## 5. Step 3: samples

| id | definition | purpose |
|---|---|---|
| R1 | last 1,964 sessions with a valid trailing premium, ending 2026-08-26 (the last VIX date in D2–D3) | **the sample behind the printed table**; it starts 2018-08-03 |
| R2 | R1 with the forward window | robustness |
| R3 | R1 with intraday RV | robustness |
| R4 | 2020-01-01 → 2026-08-26, trailing | the period the draft *states* |
| R5 | R4 with official-close returns (D3) | close-definition check |
| R6 | 2017-05 → 2026-08-26 | all available data |

## 6. Step 4: statistics (exact definitions)

- **Share non-positive**: the fraction of sessions with $\hat\lambda_t\le0$.
- **Episode**: a maximal run of consecutive valid sessions with $\hat\lambda_t\le0$.
- **Minimum**: the smallest value, with its date.
- **Table 4**: for threshold $h\in\{0.02,0.05,0.10,0.20\}$, $n_h=\#\{t:\hat\lambda_t<-h,\ t+1\text{ valid}\}$, $k_h=\#\{\dots,\ \hat\lambda_{t+1}<-2h\}$ and $p_h=k_h/n_h$.

Code: `diagnostics.conditional_table`, `realdata.stats`.

## 7. Step 5: model-vs-data tests

| test | statistic | benchmark | verdict rule |
|---|---|---|---|
| P1 persistence | AR(1) $\hat\phi$, $\kappa=-252\log\hat\phi$ | $\phi<1$ | consistent if $\hat\phi<1$ (dominated by 2020: $\hat\phi=0.930$ without 2020) |
| P1 linear drift | OLS of $\Delta\hat\lambda_{t+1}$ on $\hat\lambda_t,\hat\lambda_t^2,\min(\hat\lambda_t,0)$, Newey–West (25 lags) | zero loadings | rejected if either $\lvert t\rvert>1.96$ |
| P2 Gaussian law | skewness, excess kurtosis, Jarque–Bera | 0, 0, $\chi^2_2$ | rejected if JB > 5.99 |
| P2 share | $\Phi(-\bar\lambda/s)$ vs observed share | equal | rejected if they differ by > 0.05 |
| P6 minimum | rank of the observed minimum among 2,000 simulated minima of the fitted Gaussian AR(1) | 5–95% band | rejected outside the band |
| P4 skewness sign | sample skewness | corrected Section 8: negative | sign comparison |
| P5 mean | mean, Newey–West (42 lags) | > 0 | consistent if $t>1.96$ |

Code: `realdata.verdicts`, `realdata.fitted_ou_benchmark`.

## 8. Step 6: ex-ante premium (P3)

- **Target**: $RV^{\rm fwd}_t$.
- **Predictors**: the HAR daily, weekly and monthly averages of $252\,(rv5+(r^{on})^2)$; for the US, $252\,r^2$.
- **Variants**: levels; logs (conditional-mean forecast); logs-median (conditional-median forecast); levels plus $\mathrm{IV}^2$.
- **Estimation**: expanding window with at least 500 pairs, refitted every 21 sessions. At date $t$ only pairs with $s\le t-21$ are used, because $RV^{\rm fwd}_s$ is known only at $s+21$.
- **Reported calibration**: out-of-sample $R^2$, median forecast error, and the share of $RV<F$.
- **Outcome**: the share of $\hat\pi_t\le0$, the minimum and the longest run.

Code: `realdata.har_exante`.

## 9. Step 7: mechanism (U1, U2)

- **Deployment proxies**:
  - detrended log total short index-option OI;
  - (client + pro net short) / total short;
  - FII net short / total short;
  - detrended log vega outstanding.
- **Aggregation**: Friday-ending weekly means, to average out the expiry cycle.
- **M1**: correlation and Newey–West $t$ (8 lags) of $\mathrm{IV}^2$ and $\hat\lambda$ on each proxy, in levels and in weekly changes. Assumption 2.1 with a constant intercept implies correlation −1.
- **M2**: weekly AR(1) speeds of $\hat\lambda$ and of each proxy. The model gives them the same κ.

Code: `userfiles.py`, `realdata.mechanism_table`.

## 10. Step 8: outputs

```bash
python scripts/08_real_data_validation.py
```

| file | content |
|---|---|
| `results/08_real_data_validation.md` | full report with the audit |
| `results/08_reproduction.csv` | Table 4 and statistics for R1–R6 next to the printed values |
| `results/08_model_vs_data.csv` | P1–P6 for India (R1, R4) and the US |
| `results/08_exante.csv`, `08_mechanism.csv`, `08_speeds.csv` | layers C1 and C2 |
| `paper/table_*.tex` | LaTeX tables generated from the same numbers |
| `paper/section10_generated.tex` | Section 10 prose with every number inserted by code |
| `results/08_india_premium.png` | premium path and log-density against the Gaussian |

## 11. Verifier's checklist (values from the pinned data)

| quantity (R1) | value |
|---|---|
| first / last session | 2018-08-03 / 2026-08-26 |
| non-positive sessions | 339 (17.3%) |
| episodes / longest | 51 / 28 |
| minimum | −0.5300 on 2020-04-09 (−0.5348 with official closes) |
| Table 4 $n_h$ | 90, 46, 22, 15 |
| Table 4 $p_h$ | 0.544, 0.457, 0.682, 0.467 |

The printed draft has n 90/46/22/15 and p 0.556/0.457/0.682/0.467, longest 28, minimum −0.535. The remaining differences come from:
- one session at $h=0.02$ (49 vs 50 of 90);
- the close definition;
- 16 days missing from Yahoo's VIX, which Kite data would fill.

## 12. Known limitations

- Yahoo's India VIX has a few gaps and repeated values. NSE or Kite data remove this.
- The one-minute close is a proxy for the official close before 2020.
- Weekly deployment proxies measure contracts, not rupee capital, and include hedgers on both sides.
- None of the layers identifies the clearing slope $b$. That requires exogenous supply shifts, for example margin-rule changes.
