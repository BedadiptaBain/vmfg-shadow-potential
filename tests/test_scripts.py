"""The symbolic audit runs clean (all checks, including negative controls)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_script_passes():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "01_symbolic_verification.py")],
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-2000:]
