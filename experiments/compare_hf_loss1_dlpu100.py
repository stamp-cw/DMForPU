"""Compare HF epsilon+loss1 (best of first 400 epochs) with DLPU (first 100).

The HF checkpoint is selected only from the first 400-epoch validation history
using U3-aligned NRMSE.  Inference steps are swept on the validation split;
the selected step count is then used for the held-out GFS128 test conditions.
DLPU is selected from its first 100 epochs by the same validation metric.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_gfs128_t200_prediction_study as hf_study
from experiments import train_gfs128_upstream as upstream

OUT = ROOT / "experiments/results/gfs128_t200_prediction_study/hf_loss1_vs_dlpu100"
HF_VARIANT = "hf_epsilon_loss1"
STEPS = hf_study.INFERENCE_STEPS
CONDITIONS = ("clean", "0", "5", "10", "20", "30")
METRICS = ("raw_mae", "raw_rmse", "mean_aligned_mae", "mean_aligned_rmse",
           "mean_aligned_nrmse", "u3_aligned_mae", "u3_aligned_rmse",
           "u3_aligned_nrmse", "u3_aligned_ssim", "pge",
           "rewrap_circular_mae", "range_aligned_au")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_h5(condition: str):
    name = "test_clean.h5" if condition == "clean" else f"test_{condition}dB.h5"
    with h5py.File(ROOT / "data/GFS128" / name, "r") as f:
        return tuple(torch.from_numpy(f[key][:]) for key in ("psi", "phi", "snr"))


def load_hf_best_first400():
    run = hf_study.RUNS / HF_VARIANT
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    valid = [r for r in history if int(r["epoch"]) <= 400 and r.get("validated")
             and r.get("val_u3_aligned_nrmse") is not None]
    best = min(valid, key=lambda r: r["val_u3_aligned_nrmse"])
    epoch = int(best["epoch"])
    model = hf_study.build_model(HF_VARIANT).eval()
    state = torch.load(run / "weights" / f"epoch_{epoch:03d}.pth",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model, epoch, best


def load_dlpu_best_first100():
    run = ROOT / "experiments/results/gfs128_upstream/runs/dlpu"
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    valid = [r for r in history if int(r["epoch"]) <= 100
             and r.get("val_u3_aligned_nrmse") is not None]
    best = min(valid, key=lambda r: r["val_u3_aligned_nrmse"])
    epoch = int(best["epoch"])
    model = upstream.build("dlpu").eval()
    state = torch.load(run / "weights" / f"epoch_{epoch:03d}.pth",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model, epoch, best


def split_from_manifest():
    _, val, _ = upstream.datasets()
    return val


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hf_model, hf_epoch, hf_val = load_hf_best_first400()
    dlpu_model, dlpu_epoch, dlpu_val = load_dlpu_best_first100()
    _, validation = hf_study.load_train_val()

    sweep = []
    for steps in STEPS:
        row = {"method": "hf_epsilon_loss1", "split": "validation",
               "epoch": hf_epoch, "steps": steps, "seed": 21000}
        row.update(hf_study.evaluate(hf_model, validation, steps, 21000, timing=True))
        sweep.append(row)
        write_csv(OUT / "hf_validation_step_sweep.csv", sweep)
        print(f"HF epoch={hf_epoch} steps={steps} U3-NRMSE={row['u3_aligned_nrmse']:.8f}", flush=True)
    hf_best_step = min(sweep, key=lambda r: r["u3_aligned_nrmse"])["steps"]

    # Validation comparison at the same data split, then held-out conditions.
    val_rows = []
    hf_val_eval = next(r for r in sweep if r["steps"] == hf_best_step)
    val_rows.append({"method": "hf_epsilon_loss1", "epoch": hf_epoch,
                     "steps": hf_best_step, **{k: hf_val_eval[k] for k in METRICS}})
    dlpu_val_split = (validation[0], validation[1], validation[2])
    # Upstream and diffusion adapters use identical tensor split conventions.
    dlpu_val_metrics = upstream.evaluate(dlpu_model, "dlpu", dlpu_val_split, upstream.BATCH["dlpu"])
    val_rows.append({"method": "dlpu", "epoch": dlpu_epoch, "steps": 1,
                     **{k: dlpu_val_metrics[k] for k in METRICS}})
    write_csv(OUT / "validation_comparison.csv", val_rows)

    test_rows = []
    for condition in CONDITIONS:
        split = load_h5(condition)
        hf_metrics = hf_study.evaluate(hf_model, tuple(x[:, None] for x in split),
                                       hf_best_step, 30000 + CONDITIONS.index(condition), timing=True)
        test_rows.append({"method": "hf_epsilon_loss1", "condition": condition,
                          "epoch": hf_epoch, "steps": hf_best_step, **hf_metrics})
        dlpu_metrics = upstream.evaluate(dlpu_model, "dlpu", tuple(x[:, None] for x in split),
                                         upstream.BATCH["dlpu"])
        test_rows.append({"method": "dlpu", "condition": condition,
                          "epoch": dlpu_epoch, "steps": 1, **dlpu_metrics})
        write_csv(OUT / "test_comparison.csv", test_rows)

    summary = {
        "selection": "minimum validation U3-aligned NRMSE",
        "hf": {"variant": HF_VARIANT, "first_epochs": 400, "selected_epoch": hf_epoch,
               "selected_validation_u3_aligned_nrmse": hf_val["val_u3_aligned_nrmse"],
               "selected_steps": hf_best_step},
        "dlpu": {"first_epochs": 100, "selected_epoch": dlpu_epoch,
                 "selected_validation_u3_aligned_nrmse": dlpu_val["val_u3_aligned_nrmse"]},
        "inference_steps": list(STEPS), "validation": val_rows,
        "test": test_rows, "updated_at": time.strftime("%F %T"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot([r["steps"] for r in sweep], [r["u3_aligned_nrmse"] for r in sweep], "o-")
    axes[0].axvline(hf_best_step, color="crimson", ls="--", label=f"best={hf_best_step}")
    axes[0].set_xlabel("HF inference steps"); axes[0].set_ylabel("U3-aligned NRMSE")
    axes[0].set_title("HF loss1 validation step sweep"); axes[0].legend()
    for metric, ax, title in (("u3_aligned_nrmse", axes[1], "Test U3-aligned NRMSE"),
                              ("u3_aligned_ssim", axes[2], "Test U3-aligned SSIM")):
        for method in ("hf_epsilon_loss1", "dlpu"):
            rows = [r for r in test_rows if r["method"] == method]
            ax.plot(list(CONDITIONS), [r[metric] for r in rows], "o-", label=method)
        ax.set_title(title); ax.set_xlabel("condition"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "comparison.png", dpi=220); plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
