"""Evaluate missing HF loss1 inference step counts on the existing validation split."""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_gfs128_t200_prediction_study as study

RUN = study.RUNS / "hf_epsilon_loss1"
OUT = study.OUT / "hf_loss1_vs_dlpu100"
CSV_PATH = OUT / "hf_validation_step_sweep.csv"
STEPS = (10, 25, 50, 75, 100, 150, 200)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    if CSV_PATH.exists():
        with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    done = {int(r["steps"]) for r in rows if r.get("method") == "hf_epsilon_loss1"}
    requested = [s for s in STEPS if s not in done]
    if not requested:
        print("all requested step counts already exist", flush=True)
        return

    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    valid = [r for r in history if int(r["epoch"]) <= 400 and r.get("validated")
             and r.get("val_u3_aligned_nrmse") is not None]
    best = min(valid, key=lambda r: r["val_u3_aligned_nrmse"])
    epoch = int(best["epoch"])
    model = study.build_model("hf_epsilon_loss1").eval()
    state = torch.load(RUN / "weights" / f"epoch_{epoch:03d}.pth",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    _, validation = study.load_train_val()

    fieldnames = None
    for steps in requested:
        metrics = study.evaluate(model, validation, steps, 21000, timing=True)
        row = {"method": "hf_epsilon_loss1", "split": "validation",
               "epoch": epoch, "steps": steps, "seed": 21000, **metrics}
        rows = [r for r in rows if not (r.get("method") == "hf_epsilon_loss1"
                                       and int(r.get("steps", -1)) == steps)]
        rows.append(row)
        fieldnames = list(row)
        with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        print(f"epoch={epoch} steps={steps} U3-NRMSE={metrics['u3_aligned_nrmse']:.8f} "
              f"SSIM={metrics['u3_aligned_ssim']:.8f}", flush=True)
    print(CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
