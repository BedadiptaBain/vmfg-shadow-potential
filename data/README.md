# Data for Section 10

The NIFTY / India VIX data are not redistributed here (NSE terms of use).
Place your own file in this folder (it is git-ignored) and run

    python scripts/06_empirical_section10.py --data data/DAILY.csv
    python scripts/06_empirical_section10.py --data data/Manifold_Research_Data.xlsx --sheet DAILY

## Accepted formats

One file with a date column and two numeric columns. Column names are matched
case-insensitively:

| role | accepted names |
|---|---|
| date | `date`, `timestamp`, `trade_date` (day-first parsing, e.g. 03-09-2018) |
| NIFTY close | `nifty`, `close`, `nifty_close`, `nifty50`, `nifty_50`, `price` |
| India VIX | `vix`, `india_vix`, `indiavix`, `iv`, `implied_vol` (percent or decimal) |

This matches the `DAILY` sheet (date, nifty, vix) written by the Kite backfill
in the Apps Script workbook. Alternatively pass two NSE downloads:

    python scripts/06_empirical_section10.py --index nifty50.csv --vix indiavix.csv

(both need `Date` and `Close`).

## What the loader enforces

Rows are sorted by date (a rolling window on unsorted rows is meaningless),
duplicate dates are dropped, VIX in decimals is converted to percent, and
large daily moves and calendar gaps are listed. Every fix is reported at the
top of `results/06_section10_data.md`.

## Conventions (state these in the paper)

- `IV2_t = (VIX_t/100)^2`, annualised, 30 calendar days.
- `r_t = log(close_t/close_{t-1})`, window `w = 21` sessions, annualisation 252.
- trailing proxy `lam_back_t = IV2_t - (252/21) * sum_{j=0}^{20} r_{t-j}^2`
- forward proxy `lam_fwd_t = IV2_t - (252/21) * sum_{j=1}^{21} r_{t+j}^2`
- Use `--start 2020-01-01` to reproduce a 2020-onward sample.
