"""Run the architecture-matched WWFCA-v2 screening pair concurrently."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/gfs128_wwfca_v2"
METHODS = ("wwfca_v2_off", "wwfca_v2")


def write_status(value):
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = OUT / "schedule_status.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT / "schedule_status.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    processes = {}
    streams = []
    started = time.time()
    for method in METHODS:
        stdout = (OUT / f"{method}.log").open("a", encoding="utf-8", buffering=1)
        stderr = (OUT / f"{method}.err.log").open("a", encoding="utf-8", buffering=1)
        command = [
            sys.executable, "-B", "-u", str(ROOT / "experiments/train_gfs_rme128.py"),
            "train", "--dataset", "GFS128", "--method", method,
        ]
        processes[method] = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        streams.extend((stdout, stderr))
    while any(process.poll() is None for process in processes.values()):
        jobs = {
            method: {
                "pid": process.pid,
                "state": "training" if process.poll() is None else "complete" if process.returncode == 0 else "failed",
                "exit_code": process.poll(),
            }
            for method, process in processes.items()
        }
        write_status({"state": "training", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "elapsed_seconds": time.time() - started, "epochs": 100, "jobs": jobs})
        time.sleep(20)
    for stream in streams:
        stream.close()
    jobs = {
        method: {"pid": process.pid, "state": "complete" if process.returncode == 0 else "failed",
                 "exit_code": process.returncode}
        for method, process in processes.items()
    }
    write_status({"state": "complete" if all(p.returncode == 0 for p in processes.values()) else "failed",
                  "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "elapsed_seconds": time.time() - started, "epochs": 100, "jobs": jobs})
    return 0 if all(p.returncode == 0 for p in processes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
