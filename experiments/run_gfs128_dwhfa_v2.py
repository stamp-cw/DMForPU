"""Queue DWHFA-v2 epsilon experiments after official Restormer."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/results/gfs128_t200_prediction_study/runs"
LOG = ROOT / "experiments/results/gfs128_t200_prediction_study/dwhfa_v2_schedule.log"
VARIANTS = ("dwhfa_v2_epsilon", "dwhfa_v2_epsilon_loss2")


def log(message: str):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def complete_restormer() -> bool:
    folder = ROOT / "experiments/results/gfs128_upstream/runs/restormer"
    if (folder / "complete.json").exists() or (folder / "training_complete.json").exists():
        return True
    status = folder / "status.json"
    if status.exists():
        try:
            return json.loads(status.read_text(encoding="utf-8")).get("state") == "complete"
        except Exception:
            pass
    return False


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("DWHFA-v2 schedule started; waiting for Restormer 300")
    while not complete_restormer():
        time.sleep(30)
    log("Restormer complete; starting DWHFA-v2 epsilon experiments")
    for variant in VARIANTS:
        log(f"start {variant} for 400 epochs")
        result = subprocess.run([
            sys.executable, "-B", "-u",
            str(ROOT / "experiments/train_gfs128_t200_prediction_study.py"),
            "train", "--variant", variant,
        ], cwd=ROOT)
        if result.returncode:
            log(f"FAILED {variant} exit={result.returncode}")
            raise SystemExit(result.returncode)
        log(f"complete {variant}")
    log("DWHFA-v2 schedule complete")


if __name__ == "__main__":
    main()
