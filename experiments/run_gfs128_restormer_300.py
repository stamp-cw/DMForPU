"""Queue official Restormer training for 300 epochs after current loss2 jobs."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/results/gfs128_upstream/runs"
T200_RUNS = ROOT / "experiments/results/gfs128_t200_prediction_study/runs"
LOG = ROOT / "experiments/results/gfs128_upstream/restormer_300_schedule.log"


def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def complete(name: str) -> bool:
    folder = RUNS / name
    if (folder / "complete.json").exists() or (folder / "training_complete.json").exists():
        return True
    status = folder / "status.json"
    if not status.exists():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("state") == "complete"
    except Exception:
        return False


def main():
    log("Restormer 300 schedule started; waiting for dwhfa_epsilon_loss2")
    while not ((T200_RUNS / "dwhfa_epsilon_loss2" / "training_complete.json").exists()
               or (T200_RUNS / "dwhfa_epsilon_loss2" / "status.json").exists()
               and json.loads((T200_RUNS / "dwhfa_epsilon_loss2" / "status.json").read_text(encoding="utf-8")).get("state") == "complete"):
        time.sleep(30)
    log("dwhfa_epsilon_loss2 complete; starting official Restormer for 300 epochs")
    result = subprocess.run([
        sys.executable, "-B", "-u",
        str(ROOT / "experiments/train_gfs128_upstream.py"),
        "train", "--method", "restormer", "--epochs", "300",
    ], cwd=ROOT)
    if result.returncode:
        log(f"FAILED restormer exit={result.returncode}")
        raise SystemExit(result.returncode)
    log("complete restormer 300")


if __name__ == "__main__":
    main()
