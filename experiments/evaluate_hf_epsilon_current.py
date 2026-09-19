"""Evaluate the current best HF-epsilon checkpoint at fixed 25-step inference."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_gfs128_t200_prediction_study import (
    OUT, RUNS, SELECTION_KEY, build_model, evaluate, load_train_val, read_test,
)

VARIANT = "hf_epsilon"
STEPS = 25
RUN = RUNS / VARIANT
EVAL_OUT = RUN / "current_best_epoch_evaluation"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_best():
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    valid = [row for row in history if row.get("validated") and row.get(f"val_{SELECTION_KEY}") is not None]
    best = min(valid, key=lambda row: row[f"val_{SELECTION_KEY}"])
    epoch = int(best["epoch"])
    model = build_model(VARIANT).eval()
    state = torch.load(RUN / "weights" / f"epoch_{epoch:03d}.pth",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model, history, best, epoch


@torch.inference_mode()
def prediction_examples(model, split, seed=45100, count=1):
    wrapped, target, snr = split
    wi, ti, si = wrapped[:count].cuda(), target[:count].cuda(), snr[:count].cuda()
    sigma = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., si / 10))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model.sample(wi, sigma, generator=generator, steps=STEPS)
    return wi[0, 0].float().cpu().numpy(), pred[0, 0].float().cpu().numpy(), ti[0, 0].float().cpu().numpy()


def main():
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    model, history, best, epoch = load_best()
    _, validation = load_train_val()
    rows = [{"split": "validation", "condition": "val", "epoch": epoch, "steps": STEPS,
             **evaluate(model, validation, STEPS, 45100, timing=True)}]
    for index, condition in enumerate(("clean", "0", "5", "10", "20", "30")):
        rows.append({"split": "test", "condition": condition, "epoch": epoch, "steps": STEPS,
                     **evaluate(model, read_test(condition), STEPS, 45200 + index, timing=True)})
    write_csv(EVAL_OUT / "metrics.csv", rows)
    numeric = {key: float(value) for key, value in rows[0].items()
               if isinstance(value, (float, int))}
    summary = {
        "variant": VARIANT, "checkpoint_epoch": epoch, "inference_steps": STEPS,
        "selection_metric": "validation U3-aligned NRMSE",
        "selection_value": float(best[f"val_{SELECTION_KEY}"]),
        "metrics": rows,
        "note": "Current best checkpoint; training may resume after this evaluation.",
    }
    (EVAL_OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    valid = [row for row in history if row.get("validated") and row.get(f"val_{SELECTION_KEY}") is not None]
    conditions = [row["condition"] for row in rows[1:]]
    tests = rows[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(conditions, [row["u3_aligned_nrmse"] for row in tests], color="#3465a4")
    axes[0].set_title("HF epsilon: U3-aligned NRMSE")
    axes[0].set_ylabel("NRMSE")
    axes[1].bar(conditions, [row["u3_aligned_ssim"] for row in tests], color="#4e9a06")
    axes[1].set_title("HF epsilon: U3-aligned SSIM")
    axes[1].set_ylabel("SSIM")
    axes[2].bar(conditions, [row["milliseconds_per_image"] for row in tests], color="#c17d11")
    axes[2].set_title("25-step latency")
    axes[2].set_ylabel("ms/image")
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(EVAL_OUT / "metrics.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], color="#3465a4")
    axes[0].set_yscale("log")
    axes[0].set_title("HF epsilon training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot([row["epoch"] for row in valid], [row[f"val_{SELECTION_KEY}"] for row in valid], "o-", color="#75507b")
    axes[1].axvline(epoch, color="#cc0000", linestyle="--", label=f"best epoch {epoch}")
    axes[1].set_title("25-step validation U3-NRMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(EVAL_OUT / "training_curve.png", dpi=220)
    plt.close(fig)

    example_conditions = ("clean", "0", "10", "30")
    fig, axes = plt.subplots(len(example_conditions), 4, figsize=(13, 12), squeeze=False)
    for row_index, condition in enumerate(example_conditions):
        split = read_test(condition)
        wrapped, prediction, target = prediction_examples(model, split, 46000 + row_index)
        error = np.abs(prediction - target)
        images = (wrapped, prediction, target, error)
        labels = ("wrapped input", "prediction", "target", "absolute error")
        for col, (image, label) in enumerate(zip(images, labels)):
            axis = axes[row_index, col]
            if label == "absolute error":
                display = axis.imshow(image, cmap="magma")
            else:
                display = axis.imshow(image, cmap="viridis")
            axis.set_title(f"{condition} dB\n{label}")
            axis.axis("off")
            fig.colorbar(display, ax=axis, fraction=.046, pad=.03)
    fig.suptitle(f"HF epsilon qualitative results, epoch {epoch}, {STEPS} steps")
    fig.tight_layout()
    fig.savefig(EVAL_OUT / "qualitative_examples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
