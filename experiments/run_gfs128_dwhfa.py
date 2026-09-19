"""Run DWHFA x0/epsilon studies after the queued loss1 experiments."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/gfs128_t200_prediction_study"
WAIT_FOR = ("hf_epsilon_loss1", "directional_epsilon_loss1")
VARIANTS = ("dwhfa_x0", "dwhfa_epsilon")
LOG = OUT / "dwhfa_schedule.log"

def log(msg):
    with LOG.open("a", encoding="utf-8") as f: f.write(time.strftime("[%F %T] ") + msg + "\n")

def complete(name):
    p = OUT / "runs" / name / "training_complete.json"
    try: return int(json.loads(p.read_text(encoding="utf-8")).get("epochs", 0)) >= (400 if name.endswith("loss1") else 300)
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError): return False

def main():
    log("DWHFA schedule started; waiting for loss1 experiments")
    while not all(complete(x) for x in WAIT_FOR): time.sleep(30)
    log("loss1 experiments complete; starting DWHFA")
    for variant in VARIANTS:
        if complete(variant): log(f"skip completed {variant}"); continue
        out = (OUT / f"{variant}.log").open("a", encoding="utf-8")
        err = (OUT / f"{variant}.err.log").open("a", encoding="utf-8")
        try:
            code = subprocess.call([sys.executable, "-B", "-u", str(ROOT/"experiments/train_gfs128_t200_prediction_study.py"), "train", "--variant", variant], cwd=ROOT, stdout=out, stderr=err)
        finally: out.close(); err.close()
        if code: log(f"FAILED {variant} exit={code}"); raise SystemExit(code)
        log(f"complete {variant}")
    log("DWHFA schedule complete")

if __name__ == "__main__": main()
