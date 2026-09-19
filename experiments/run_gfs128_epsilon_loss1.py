"""Run the two loss1 epsilon experiments sequentially after DLPU finishes."""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/gfs128_t200_prediction_study"
DLPU_COMPLETE = ROOT / "experiments/results/gfs128_upstream/runs/dlpu/complete.json"
VARIANTS = ("hf_epsilon_loss1", "directional_epsilon_loss1")
LOG = OUT / "loss1_schedule.log"

def log(message: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(time.strftime("[%F %T] ") + message + "\n")

def dlpu_done():
    try:
        obj = json.loads(DLPU_COMPLETE.read_text(encoding="utf-8"))
        return int(obj.get("epochs_completed", 0)) >= 300
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return False

def main():
    log("loss1 schedule started; waiting for DLPU epoch 300")
    while not dlpu_done():
        time.sleep(30)
    log("DLPU epoch 300 detected; starting loss1 variants")
    for variant in VARIANTS:
        folder = OUT / "runs" / variant
        if (folder / "training_complete.json").exists():
            log(f"skip completed {variant}")
            continue
        log(f"start {variant} for 400 epochs")
        out = (OUT / f"{variant}.log").open("a", encoding="utf-8")
        err = (OUT / f"{variant}.err.log").open("a", encoding="utf-8")
        try:
            code = subprocess.call([sys.executable, "-B", "-u", str(ROOT / "experiments/train_gfs128_t200_prediction_study.py"),
                                    "train", "--variant", variant], cwd=ROOT, stdout=out, stderr=err)
        finally:
            out.close(); err.close()
        if code:
            log(f"FAILED {variant} exit={code}")
            raise SystemExit(code)
        log(f"complete {variant}")
    log("loss1 schedule complete")

if __name__ == "__main__":
    main()
