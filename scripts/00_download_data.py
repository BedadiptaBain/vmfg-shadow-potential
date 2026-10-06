#!/usr/bin/env python3
"""00 - Download raw market data and build the canonical daily datasets.

Examples
  # reproducible route (pinned GitHub mirrors; works anywhere GitHub is reachable)
  python scripts/00_download_data.py --source github

  # official / primary routes (run on your own machine)
  python scripts/00_download_data.py --source yahoo --market india --start 2016-01-01 --end 2026-09-16
  python scripts/00_download_data.py --source yahoo --market us    --start 2005-01-01 --end 2026-09-16
  python scripts/00_download_data.py --source nse   --start 2018-01-01 --end 2026-09-16
  python scripts/00_download_data.py --source kite  --start 2018-01-01 --end 2026-09-16 \
         --kite-api-key XXXX --kite-access-token YYYY
  python scripts/00_download_data.py --source niftyindices --start 2018-01-01 --end 2026-09-16
  python scripts/00_download_data.py --source fred    # US: SP500, VIXCLS
  python scripts/00_download_data.py --source cboe    # US: VIX history

Outputs
  data/raw/<source>/...            untouched downloads
  data/raw/provenance.jsonl        url, sha256, bytes, UTC time of every file
  data/processed/india_daily_<source>.csv, us_daily_<source>.csv
  data/processed/build_audit_<source>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vmfg.marketdata import build as Bd, sources as Sr  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    choices=["github", "yahoo", "nse", "niftyindices", "kite", "fred", "cboe"])
    ap.add_argument("--market", default="india", choices=["india", "us"])
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-16")
    ap.add_argument("--kite-api-key")
    ap.add_argument("--kite-access-token")
    ap.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)

    if a.source == "github":
        files = Sr.download_github(RAW, refresh=a.refresh)
    elif a.source == "yahoo":
        files = Sr.download_yahoo(RAW, a.market, a.start, a.end, refresh=a.refresh)
    elif a.source == "nse":
        files = Sr.download_nse(RAW, a.start, a.end, refresh=a.refresh)
    elif a.source == "niftyindices":
        files = Sr.download_niftyindices(RAW, a.start, a.end, refresh=a.refresh)
    elif a.source == "kite":
        if not (a.kite_api_key and a.kite_access_token):
            sys.exit("--kite-api-key and --kite-access-token are required")
        files = Sr.download_kite(RAW, a.kite_api_key, a.kite_access_token, a.start, a.end, refresh=a.refresh)
    elif a.source == "fred":
        files = Sr.download_fred(RAW, refresh=a.refresh)
    else:
        files = Sr.download_cboe(RAW, refresh=a.refresh)
    print(f"{len(files)} raw files under {RAW}")
    if a.download_only:
        return

    built = []
    if a.source == "github":
        built.append(("india", *Bd.build_india_github(RAW)))
        built.append(("us", *Bd.build_us_github(RAW)))
    elif a.source == "yahoo":
        built.append((a.market, *Bd.build_from_yahoo(RAW, a.market)))
    elif a.source == "nse":
        built.append(("india", *Bd.build_from_nse(RAW)))
    elif a.source == "kite":
        built.append(("india", *Bd.build_from_kite(RAW)))
    else:
        print("raw files downloaded; combine them with the matching index/VIX series before analysis "
              "(see data/README.md)")
    for market, df, audit in built:
        out = PROC / f"{market}_daily_{a.source}.csv"
        df.to_csv(out, index=False, float_format="%.10g")
        (PROC / f"build_audit_{market}_{a.source}.json").write_text(json.dumps(audit, indent=1))
        print(f"{market}: {len(df)} sessions {audit.get('first_date')} .. {audit.get('last_date')} -> {out}")


if __name__ == "__main__":
    main()
