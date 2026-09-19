"""Run the v3 analyzer when the already-running training schedule finishes."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/"experiments/results/gfs128_wwfca_v3/schedule_status.json"

while True:
    if STATUS.exists():
        state=json.loads(STATUS.read_text(encoding="utf-8")).get("state")
        if state=="complete":
            raise SystemExit(subprocess.call([sys.executable,"-B",str(ROOT/"experiments/analyze_gfs128_wwfca_v3.py")],cwd=ROOT))
        if state=="failed":
            raise SystemExit(1)
    time.sleep(30)
