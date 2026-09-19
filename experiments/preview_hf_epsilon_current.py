"""Small, non intrusive preview of the current best HF-epsilon checkpoint.

This intentionally evaluates only a fixed prefix of each split so it can run
alongside the long training job without changing or stopping that job.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_gfs128_t200_prediction_study import (  # noqa: E402
    RUNS, SELECTION_KEY, build_model, evaluate, load_train_val, read_test,
)

VARIANT = "hf_epsilon"
STEPS = 25
COUNT = 100
RUN = RUNS / VARIANT
OUT = RUN / "current_best_epoch_preview"


def crop(split):
    return tuple(x[:COUNT] for x in split)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    valid = [r for r in history if r.get("validated") and r.get(f"val_{SELECTION_KEY}") is not None]
    best = min(valid, key=lambda r: r[f"val_{SELECTION_KEY}"])
    epoch = int(best["epoch"])
    model = build_model(VARIANT).eval()
    state = torch.load(RUN / "weights" / f"epoch_{epoch:03d}.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    _, val = load_train_val()
    rows = [{"split": "validation", "condition": "val", "epoch": epoch, "steps": STEPS,
             **evaluate(model, crop(val), STEPS, 65100, timing=False)}]
    for i, condition in enumerate(("clean", "0", "5", "10", "20", "30")):
        rows.append({"split": "test", "condition": condition, "epoch": epoch, "steps": STEPS,
                     **evaluate(model, crop(read_test(condition)), STEPS, 65200 + i, timing=False)})
    (OUT / "summary.json").write_text(json.dumps({
        "variant": VARIANT, "checkpoint_epoch": epoch, "inference_steps": STEPS,
        "samples_per_condition": COUNT, "selection_metric": "validation U3-aligned NRMSE",
        "selection_value": float(best[f"val_{SELECTION_KEY}"]), "metrics": rows,
        "note": "Preview subset; training was left running.",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    conditions = [r["condition"] for r in rows[1:]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(conditions, [r["u3_aligned_nrmse"] for r in rows[1:]], color="#3465a4")
    axes[0].set_title("HF epsilon preview: U3-aligned NRMSE")
    axes[1].bar(conditions, [r["u3_aligned_ssim"] for r in rows[1:]], color="#4e9a06")
    axes[1].set_title("HF epsilon preview: U3-aligned SSIM")
    axes[2].plot([r["epoch"] for r in valid], [r[f"val_{SELECTION_KEY}"] for r in valid], "o-")
    axes[2].axvline(epoch, color="r", linestyle="--", label=f"best {epoch}")
    axes[2].set_title("Validation curve (25 steps)")
    axes[2].legend()
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "metrics_and_curve.png", dpi=220); plt.close(fig)
    # One deterministic qualitative example per condition.
    fig, axes = plt.subplots(6, 4, figsize=(12, 16), squeeze=False)
    for ri, condition in enumerate(("clean", "0", "5", "10", "20", "30")):
        wrapped, target, snr = read_test(condition)
        wi, ti, si = wrapped[:1].cuda(), target[:1].cuda(), snr[:1].cuda()
        sigma = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., si / 10))
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model.sample(wi, sigma, generator=torch.Generator(device="cuda").manual_seed(66000 + ri), steps=STEPS)
        images = (wi[0, 0].float().cpu().numpy(), pred[0, 0].float().cpu().numpy(),
                  ti[0, 0].float().cpu().numpy(), np.abs(pred[0, 0].float().cpu().numpy() - ti[0, 0].float().cpu().numpy()))
        for ci, image in enumerate(images):
            ax = axes[ri, ci]
            ax.imshow(image, cmap="magma" if ci == 3 else "viridis")
            ax.axis("off")
            if ri == 0: ax.set_title(("wrapped", "prediction", "target", "abs error")[ci])
        axes[ri, 0].set_ylabel(f"{condition} dB", rotation=90)
    fig.suptitle(f"HF epsilon qualitative preview, epoch {epoch}, 25 steps")
    fig.tight_layout(); fig.savefig(OUT / "qualitative_preview.png", dpi=180); plt.close(fig)
    print(json.dumps({"checkpoint_epoch": epoch, "metrics": rows}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
