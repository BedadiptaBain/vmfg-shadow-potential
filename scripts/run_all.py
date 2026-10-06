#!/usr/bin/env python3
"""Run the whole pipeline in order: download (pinned GitHub sources), the
mathematical checks 01-07, and the real-data validation 08.
Usage:  python scripts/run_all.py [--skip-download]"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
steps = [("00_download_data.py", ["--source", "github"])]
steps += [(p.name, []) for p in sorted(HERE.glob("[0-9][0-9]_*.py")) if not p.name.startswith("00")]
for name, extra in steps:
    if name.startswith("00") and "--skip-download" in sys.argv:
        continue
    print(f"\n=== {name} ===", flush=True)
    res = subprocess.run([sys.executable, str(HERE / name), *extra])
    if res.returncode != 0:
        sys.exit(f"{name} failed with exit code {res.returncode}")
print("\nDone. Reports: results/  LaTeX: paper/  data: data/processed/")
