# Your own exports

Put the files written by your Apps Script / Kite pipeline here:

- `participant_oi.csv`: NSE participant-wise index-option OI, with columns date, client_long, client_short, dii_long, dii_short, fii_long, fii_short, pro_long, pro_short, net_short_pro, net_short_fii, net_short_client, …
- `chain_history.csv`: daily option-chain summary, with columns date, expiry, dte, spot, total_OI, atm_iv, vega_outstanding_per_volpt, …

`scripts/08_real_data_validation.py` uses them for the mechanism test (layer C2 in EMPIRICAL_PROTOCOL.md). They are git-ignored.
