"""Start DWHFA PEA-loss training after the queued HF and directional jobs."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/results/gfs128_t200_prediction_study/runs"
LOG = ROOT / "experiments/results/gfs128_t200_prediction_study/loss2_schedule.log"


def log(message: str):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def complete(name: str) -> bool:
    p = RUNS / name
    if (p / "training_complete.json").exists():
        return True
    status = p / "status.json"
    if not status.exists():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("state") == "complete"
    except Exception:
        return False


def main():
    log("DWHFA loss2 watcher started; waiting for directional loss2")
    while not complete("directional_epsilon_loss2"):
        time.sleep(30)
    log("directional loss2 complete; starting dwhfa_epsilon_loss2 for 400 epochs")
    result = subprocess.run([
        sys.executable, str(ROOT / "experiments/train_gfs128_t200_prediction_study.py"),
        "train", "--variant", "dwhfa_epsilon_loss2"], cwd=ROOT)
    if result.returncode:
        log(f"FAILED dwhfa_epsilon_loss2 exit={result.returncode}")
        raise SystemExit(result.returncode)
    log("complete dwhfa_epsilon_loss2")


if __name__ == "__main__":
    main()
