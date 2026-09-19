"""Train and analyze the GFS128 validate-then-retrieve v4.3 experiment."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "gfs128_wwfca_v43"
METHOD = "wwfca_v43"


def status(value):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "schedule_status.json"
    tmp = OUT / f"status.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stdout = (OUT / f"{METHOD}.log").open("a", encoding="utf-8", buffering=1)
    stderr = (OUT / f"{METHOD}.err.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        [sys.executable, "-B", "-u", str(ROOT / "experiments" / "train_gfs_rme128.py"),
         "train", "--dataset", "GFS128", "--method", METHOD],
        cwd=ROOT, stdout=stdout, stderr=stderr,
    )
    while process.poll() is None:
        status({"state": "training", "active": METHOD, "pid": process.pid, "epochs": 35,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": time.time() - started})
        time.sleep(20)
    stdout.close()
    stderr.close()
    if process.returncode:
        status({"state": "failed", "exit_code": process.returncode})
        return process.returncode
    code = subprocess.call([sys.executable, "-B", str(ROOT / "experiments" / "analyze_gfs128_wwfca_v43.py")], cwd=ROOT)
    status({"state": "complete", "active": None, "exit_code": 0, "analysis_exit_code": code,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": time.time() - started})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
