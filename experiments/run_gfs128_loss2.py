"""Queue the two PEA-loss ablations after the currently running DWHFA job."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/results/gfs128_t200_prediction_study/runs"
LOG = ROOT / "experiments/results/gfs128_t200_prediction_study/loss2_schedule.log"
VARIANTS = ("hf_epsilon_loss2", "directional_epsilon_loss2")


def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] {message}\n")


def state(name: str):
    p = RUNS / name / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    log("loss2 schedule started; waiting for DWHFA epsilon")
    while True:
        st = state("dwhfa_epsilon")
        complete_marker = RUNS / "dwhfa_epsilon" / "training_complete.json"
        if complete_marker.exists() or (st and st.get("state") == "complete"):
            break
        time.sleep(30)
    log("DWHFA epsilon complete; starting loss2 variants")
    python = sys.executable
    for variant in VARIANTS:
        log(f"start {variant} for 400 epochs")
        result = subprocess.run(
            [python, str(ROOT / "experiments/train_gfs128_t200_prediction_study.py"),
             "train", "--variant", variant], cwd=ROOT
        )
        if result.returncode:
            log(f"FAILED {variant} exit={result.returncode}")
            raise SystemExit(result.returncode)
        log(f"complete {variant}")
    log("loss2 schedule complete")


if __name__ == "__main__":
    main()
