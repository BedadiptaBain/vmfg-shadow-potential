# vmfg-shadow-potential

Replication and verification code for

> B. Bain, *Equilibrium premium dynamics in a variance-supply mean-field game: a shadow-potential reduction*, manuscript, 2026.

The repository does three things.

1. **It verifies the mathematics** of the paper, symbolically (SymPy) and numerically, by independent computations
   that do not rely on the paper's closed forms.
2. **It downloads the market data** used in Section 13 from pinned sources and records the source, size and
   SHA-256 hash of every file. The raw data are **not** redistributed here (see *Data* below).
3. **It recomputes every empirical number** in Section 13 and every numerical check in Appendix B from those files.

## Quick start

```bash
git clone https://github.com/<BedadiptaBain>/vmfg-shadow-potential.git
cd vmfg-shadow-potential
pip install -e ".[test]"
pytest -q                                              # the test suite, a few seconds
python scripts/00_download_data.py --source github     # market data from pinned sources (about 60 MB)
python scripts/run_all.py                              # every check and table, about a minute after the download
```

Outputs are written to `results/` (reports, figures, CSV) and `paper/` (generated LaTeX tables). Both folders are
empty in the repository and are filled by the scripts.

## Where each result of the paper is checked

| paper | what is checked | script |
|---|---|---|
| Sections 4–8 | every algebraic step of the reduction, the shadow potential, the closed forms and the cross-section, symbolically, with negative controls | `01_symbolic_verification.py` |
| Table 2 | closed forms by independent root finding, with 50-digit residuals | `02_tables.py` |
| Theorem 6.3, Proposition 7.1 | Riccati best response and mean-field fixed point computed without the shadow potential; planner equivalence; exact payoffs; optimality under perturbations | `03_equilibrium_check.py` |
| Proposition 8.1, Theorem 6.4 | a particle system of 300 sellers using the empirical mean in the feedback | `04_particles.py` |
| Theorem 7.3, Lemma 7.4 | nonlinear clearing: planner optimum equals the mean-field fixed point, uniqueness from three starts, slope bounds | `25_general_clearing.py` |
| Remark 7.5 | constant curvature is a consequence: against any premium path the constant-curvature feedback satisfies the seller's first-order condition | `25a_curvature_identity.py` |
| Section 9 | two-factor equilibrium: symbolic verification, policy iteration, identities, temporal aggregation | `13_two_factor.py`, `15_verify_two_factor.py` |
| Propositions 10.2, 10.4 | best-response adaptation and the low-pass sensitivity of the regulator | `20_control_checks.py` |
| Theorem 11.1, Table 3, Proposition 11.2 | exact N-player Nash equilibrium, independent best response, O(1/N) rates and coupling; open-loop premium | `21_nplayer.py`, `25_general_clearing.py` |
| Propositions 12.3, 12.4 | restoring force, recovery paths, half-life, overshoot, first passage | `23_capacity_checks.py`, `24_recovery_speed.py` |
| Proposition 14.2 | stationary skewness under curved clearing: quadrature, a non-perturbative solver and Monte Carlo | `05_skewness.py` |
| Appendix B | grid refinement, Richardson extrapolation and Monte Carlo errors for every numerical method | `10_numerical_errors.py` |
| Section 13, Table 4 | premium proxies, local transition tests, inference, the two-factor fit, latent-state estimation, identification of the price impact | `08`, `11`, `12`, `13b`, `14`, `14b`, `16`–`19` |
| Section 13, Table 5 | the SEBI natural experiment: event effects, difference-in-differences, placebo inference | `22_natural_experiment.py` |

`run_all.py` runs every script whose name begins with two digits and an underscore; `13b`, `14b` and `25a` are run on
their own. Scripts `06` and `07` are auxiliary: `06` checks the author's own data exports, and `07` studies on synthetic
markets whether the empirical statistics can identify the support of the ex-ante premium.

## Data

The raw data are not redistributed, because their providers' terms do not permit it. `scripts/00_download_data.py`
fetches them: one-minute NIFTY 50 prices and the India VIX from a pinned public mirror or from NSE and
niftyindices.com, participant-wise index-option open interest from NSE, SPY from Yahoo Finance and the CBOE VIX from
CBOE. For every file it records the URL, size, SHA-256 hash and download time, so that two replications can be
compared byte for byte. A Zerodha Kite Connect download is also supported for users with their own credentials,
which are passed on the command line and never stored:

```bash
python scripts/00_download_data.py --source kite --kite-api-key KEY --kite-access-token TOKEN --start 2018-01-01
python scripts/00_download_data.py --source nse --start 2018-01-01
```

Accepted columns and conventions for your own exports are listed in [data/README.md](data/README.md), and the
empirical procedure is described step by step in [EMPIRICAL_PROTOCOL.md](EMPIRICAL_PROTOCOL.md).

## Layout

```
src/vmfg/model.py        closed forms (every symbol defined in the module docstring)
src/vmfg/lq_check.py     independent Riccati, fixed-point and Lyapunov checks
src/vmfg/numerics.py     stationary densities, the nonlinear HJB solver, numerical error control
src/vmfg/particles.py    particle and aggregate Monte Carlo
src/vmfg/marketdata/     downloads (sources.py), processing rules (build.py), your own exports (userfiles.py)
src/vmfg/empirical/      premium proxies, diagnostics, identification, real-data validation
scripts/                 numbered scripts and run_all.py
tests/                   pytest suite
results/, paper/         generated outputs (empty until the scripts are run)
data/                    raw/ and processed/ are created by 00 and are not committed; user/ is for your exports
```

## Notation

Variable names in the code follow an earlier draft of the paper. The docstring of `src/vmfg/model.py` defines every
variable, and the table above maps each script to the final manuscript.

## Citation and licence

See [CITATION.cff](CITATION.cff). The code is released under the MIT licence.
