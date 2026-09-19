"""Run the focused GFS128 study concurrently and analyze it when complete."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/gfs128_dlpu_hf_wwfca"
JOBS = {
    "dlpu": [sys.executable, "-B", "-u", str(ROOT / "experiments/train_gfs128_upstream.py"), "train", "--method", "dlpu"],
    "hf_matched": [sys.executable, "-B", "-u", str(ROOT / "experiments/train_gfs_rme128.py"), "train", "--dataset", "GFS128", "--method", "hf_matched"],
    "wwfca_only": [sys.executable, "-B", "-u", str(ROOT / "experiments/train_gfs_rme128.py"), "train", "--dataset", "GFS128", "--method", "wwfca_only"],
}


def dump(value):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "schedule_status.json.tmp"
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(OUT / "schedule_status.json")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    processes = {}
    files = []
    started = time.time()
    for method, command in JOBS.items():
        log = (OUT / f"{method}.log").open("a", encoding="utf-8", buffering=1)
        err = (OUT / f"{method}.err.log").open("a", encoding="utf-8", buffering=1)
        log.write(f"\n=== fresh study {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=err)
        processes[method] = process; files.extend((log, err))
    while True:
        states = {method: {"pid": process.pid, "state": "training" if process.poll() is None else
                           "complete" if process.returncode == 0 else "failed", "exit_code": process.poll()}
                  for method, process in processes.items()}
        dump({"state": "training" if any(x["state"] == "training" for x in states.values()) else
                       "failed" if any(x["state"] == "failed" for x in states.values()) else "analyzing",
              "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_seconds": time.time()-started,
              "scope": ["dlpu", "hf_matched", "wwfca_only"], "dcc": False, "jobs": states})
        if all(process.poll() is not None for process in processes.values()): break
        time.sleep(20)
    for file in files: file.close()
    failed = [method for method, process in processes.items() if process.returncode]
    if failed:
        dump({"state": "failed", "failed": failed, "elapsed_seconds": time.time()-started})
        return 1
    code = subprocess.call([sys.executable, "-B", str(ROOT / "experiments/analyze_gfs128_dlpu_hf_wwfca.py")], cwd=ROOT)
    dump({"state": "complete" if code == 0 else "analysis_failed", "analysis_exit_code": code,
          "elapsed_seconds": time.time()-started, "scope": list(JOBS), "dcc": False})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
