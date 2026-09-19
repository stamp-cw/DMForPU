"""Summarize and visualize the focused DLPU/HF/WWFCA GFS128 study."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.train_gfs128_upstream as upstream
import experiments.train_gfs_rme128 as diffusion

OUT = ROOT / "experiments/results/gfs128_dlpu_hf_wwfca"
UP = ROOT / "experiments/results/gfs128_upstream/runs/dlpu"
DIFF = ROOT / "experiments/results/gfs_rme128/runs/GFS128"
METHODS = ("dlpu", "hf_matched", "wwfca_only")
LABELS = {"dlpu": "DLPU", "hf_matched": "HF diffusion", "wwfca_only": "WWFCA-only diffusion"}
CONDITIONS = ("clean", "0", "5", "10", "20", "30")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_h5(condition: str):
    name = "test_clean.h5" if condition == "clean" else f"test_{condition}dB.h5"
    with h5py.File(ROOT / "data/GFS128" / name, "r") as f:
        return tuple(torch.from_numpy(f[key][:]) for key in ("psi", "phi", "snr"))


def evaluate_selected():
    results = {}
    dlpu_state = torch.load(UP / "best.pth", map_location="cpu", weights_only=False)
    dlpu = upstream.build("dlpu")
    dlpu.load_state_dict(dlpu_state["model"])
    item = {"method": "dlpu", "selected_epoch": dlpu_state["epoch"] + 1,
            "selection_metric": "u3_aligned_nrmse",
            "parameters": sum(p.numel() for p in dlpu.parameters()), "tests": {}}
    for condition in CONDITIONS:
        w, t, s = load_h5(condition)
        item["tests"][condition] = upstream.evaluate(dlpu, "dlpu", (w[:, None], t[:, None], s), upstream.BATCH["dlpu"])
    results["dlpu"] = item
    del dlpu
    torch.cuda.empty_cache()

    for method in METHODS[1:]:
        complete = json.loads((DIFF / method / "complete.json").read_text(encoding="utf-8"))
        results[method] = complete
    dump(OUT / "selected_results.json", results)
    return results


def histories():
    paths = {"dlpu": UP / "history.json", **{m: DIFF / m / "history.json" for m in METHODS[1:]}}
    return {method: json.loads(path.read_text(encoding="utf-8")) for method, path in paths.items()}


def write_best_epochs_by_metric(history):
    """Index the retained epoch checkpoints using every available validation metric."""
    maximize = {"val_u3_aligned_ssim", "val_raw_au", "val_integer_aligned_au", "val_range_aligned_au"}
    rows = []
    index = {}
    checkpoint_roots = {"dlpu": UP / "weights", **{m: DIFF / m / "weights" for m in METHODS[1:]}}
    for method, method_history in history.items():
        metric_names = sorted(key for key in method_history[0] if key.startswith("val_"))
        index[method] = {}
        for metric in metric_names:
            available = [row for row in method_history if row.get(metric) is not None]
            if not available:
                continue
            direction = "max" if metric in maximize else "min"
            selected = (max if direction == "max" else min)(available, key=lambda row: row[metric])
            checkpoint = checkpoint_roots[method] / f"epoch_{selected['epoch']:03d}.pth"
            item = {"method": LABELS[method], "metric": metric.removeprefix("val_"),
                    "direction": direction, "epoch": selected["epoch"], "value": selected[metric],
                    "checkpoint": str(checkpoint.relative_to(ROOT))}
            rows.append(item)
            index[method][item["metric"]] = {key: item[key] for key in ("direction", "epoch", "value", "checkpoint")}
    dump(OUT / "best_epochs_by_metric.json", index)
    with (OUT / "best_epochs_by_metric.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_tables(results):
    metrics = ("mean_aligned_mae", "mean_aligned_rmse", "mean_aligned_nrmse",
               "u3_aligned_mae", "u3_aligned_rmse", "u3_aligned_nrmse", "u3_aligned_ssim",
               "raw_au", "integer_aligned_au", "range_aligned_au", "pge", "rewrap_circular_mae")
    rows = []
    for method in METHODS:
        for condition in CONDITIONS:
            values = results[method]["tests"][condition]
            rows.append({"method": LABELS[method], "condition": condition,
                         "selected_epoch": results[method]["selected_epoch"],
                         "parameters": results[method]["parameters"],
                         **{key: values[key] for key in metrics}})
    with (OUT / "test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    ablation = []
    for condition in CONDITIONS:
        hf, wwf = results["hf_matched"]["tests"][condition], results["wwfca_only"]["tests"][condition]
        ablation.append({"condition": condition,
                         **{f"wwfca_minus_hf_{key}": wwf[key] - hf[key] for key in metrics}})
    with (OUT / "wwfca_ablation_delta.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation[0])); writer.writeheader(); writer.writerows(ablation)


def plot_training(history):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for method, rows in history.items():
        epochs = [r["epoch"] for r in rows]
        train_key = "train_loss" if method == "dlpu" else "train_total"
        axes[0, 0].plot(epochs, [r[train_key] for r in rows], label=LABELS[method])
        axes[0, 1].plot(epochs, [r["val_mean_aligned_mae"] for r in rows], label=LABELS[method])
        axes[1, 0].plot(epochs, [r["seconds"] for r in rows], label=LABELS[method])
        axes[1, 1].plot(epochs, [r["samples_per_second"] for r in rows], label=LABELS[method])
    labels = (("Training loss", "loss"), ("Validation mean-aligned MAE", "rad"),
              ("Epoch time", "seconds"), ("Training throughput", "images/s"))
    for axis, (title, ylabel) in zip(axes.flat, labels):
        axis.set_title(title); axis.set_xlabel("epoch"); axis.set_ylabel(ylabel)
        axis.grid(alpha=.25); axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=220); plt.close(fig)


def plot_test(results):
    x = np.arange(len(CONDITIONS))
    fields = (("mean_aligned_mae", "Mean-aligned MAE (rad)"),
              ("mean_aligned_nrmse", "Mean-aligned NRMSE"),
              ("u3_aligned_ssim", "U3-aligned SSIM"), ("range_aligned_au", "RA-AU (%)"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for method in METHODS:
        for axis, (field, _) in zip(axes.flat, fields):
            axis.plot(x, [results[method]["tests"][c][field] for c in CONDITIONS], marker="o", label=LABELS[method])
    for axis, (_, ylabel) in zip(axes.flat, fields):
        axis.set_xticks(x, CONDITIONS); axis.set_xlabel("condition / dB"); axis.set_ylabel(ylabel)
        axis.grid(alpha=.25); axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "test_comparison.png", dpi=220); plt.close(fig)


@torch.no_grad()
def qualitative_gallery():
    dlpu_state = torch.load(UP / "best.pth", map_location="cpu", weights_only=False)
    models = {"dlpu": upstream.build("dlpu")}
    models["dlpu"].load_state_dict(dlpu_state["model"])
    for method in METHODS[1:]:
        model, _ = diffusion.diffusion_model(method)
        state = torch.load(DIFF / method / "best.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); models[method] = model.eval()

    selected = (("clean", 0), ("0", 1), ("10", 2), ("30", 3))
    fig, axes = plt.subplots(len(selected), 5, figsize=(15, 11))
    titles = ("Wrapped", "Ground truth", *[LABELS[m] for m in METHODS])
    for row, (condition, index) in enumerate(selected):
        w, target, snr = load_h5(condition); wi=w[index:index+1, None].cuda(); ti=target[index]
        predictions = []
        predictions.append(models["dlpu"](wi).float().cpu()[0, 0])
        sigma = torch.zeros(1, device="cuda") if condition == "clean" else torch.sqrt(torch.tensor(10**.1, device="cuda") / torch.pow(10., snr[index:index+1].cuda()/10))
        for offset, method in enumerate(METHODS[1:]):
            generator = torch.Generator(device="cuda").manual_seed(50000 + row * 10 + offset)
            predictions.append(models[method].sample(wi, sigma, generator).cpu()[0, 0])
        images = [w[index], ti, *predictions]
        vmin, vmax = float(ti.min()), float(ti.max())
        for col, image in enumerate(images):
            if col >= 2:
                image = image - image.mean() + ti.mean()
            axes[row, col].imshow(image, cmap="turbo", vmin=-math.pi if col == 0 else vmin,
                                  vmax=math.pi if col == 0 else vmax)
            axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title(titles[col])
        axes[row, 0].set_ylabel(condition, rotation=0, labelpad=28, va="center")
    fig.tight_layout(); fig.savefig(OUT / "qualitative_comparison.png", dpi=220); plt.close(fig)
    del models
    torch.cuda.empty_cache()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = evaluate_selected(); history = histories(); write_tables(results); write_best_epochs_by_metric(history)
    plot_training(history); plot_test(results); qualitative_gallery()
    dump(OUT / "complete.json", {"state": "complete", "methods": list(METHODS),
                                  "outputs": ["selected_results.json", "test_metrics.csv",
                                              "best_epochs_by_metric.json", "best_epochs_by_metric.csv",
                                              "wwfca_ablation_delta.csv", "training_curves.png",
                                              "test_comparison.png", "qualitative_comparison.png"]})
    print(OUT)


if __name__ == "__main__":
    main()
