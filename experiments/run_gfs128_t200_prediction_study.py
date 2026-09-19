"""Sequential runner for the T=200 prediction-target study."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "gfs128_t200_prediction_study"
SCRIPT = ROOT / "experiments" / "train_gfs128_t200_prediction_study.py"
VARIANTS = ("hf_x0", "directional_x0", "hf_epsilon", "directional_epsilon")
EPOCHS = {"hf_x0": 300, "directional_x0": 300,
          "hf_epsilon": 600, "directional_epsilon": 600}


def status(value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "schedule_status.json"
    tmp = OUT / f"status.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def run_logged(arguments, name):
    stdout = (OUT / f"{name}.log").open("a", encoding="utf-8", buffering=1)
    stderr = (OUT / f"{name}.err.log").open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen([sys.executable, "-B", "-u", str(SCRIPT), *arguments],
                               cwd=ROOT, stdout=stdout, stderr=stderr)
    return process, stdout, stderr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for index, variant in enumerate(VARIANTS, 1):
        process, stdout, stderr = run_logged(["train", "--variant", variant], variant)
        while process.poll() is None:
            status({"state": "sequential_training", "active": variant,
                    "job": f"{index}/{len(VARIANTS)}", "pid": process.pid,
                    "epochs": EPOCHS[variant], "updated_at": time.strftime("%F %T"),
                    "elapsed_seconds": time.time() - started})
            time.sleep(20)
        stdout.close(); stderr.close()
        if process.returncode:
            status({"state": "failed", "active": variant, "returncode": process.returncode,
                    "updated_at": time.strftime("%F %T")})
            return process.returncode

    process, stdout, stderr = run_logged(["finalize"], "fixed_step_finalization")
    while process.poll() is None:
        status({"state": "fixed_step_finalization", "active": "checkpoint selection and sequential final tests",
                "pid": process.pid, "updated_at": time.strftime("%F %T"),
                "elapsed_seconds": time.time() - started})
        time.sleep(20)
    stdout.close(); stderr.close()
    state = "complete" if process.returncode == 0 else "failed"
    status({"state": state, "active": None, "returncode": process.returncode,
            "updated_at": time.strftime("%F %T"), "elapsed_seconds": time.time() - started})
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
