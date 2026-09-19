"""Plot the completed official-source U3Net training history on GFS128."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/results/gfs128_upstream/runs/u3net"
OUT = ROOT / "experiments/results/gfs128_upstream/figures"


def main() -> None:
    history = json.loads((RUN / "history.json").read_text(encoding="utf-8"))
    epoch = np.asarray([row["epoch"] for row in history])
    loss = np.asarray([row["train_loss"] for row in history])
    val = np.asarray([row["val_mean_aligned_mae"] for row in history])
    best = int(np.argmin(val))
    split = 500

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(12.2, 7.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.02, 1))
    ax_all = fig.add_subplot(grid[0, :])
    ax_sr = fig.add_subplot(grid[1, 0])
    ax_sd = fig.add_subplot(grid[1, 1])

    blue, orange, green = "#2878B5", "#D7632C", "#238B57"
    ax_all.plot(epoch[:split], loss[:split], color=blue, lw=1.6, label=r"Self-recovery $L_{sr}$")
    ax_all.plot(epoch[split:], loss[split:], color=orange, lw=1.6, label=r"Self-distillation $L_{sd}$")
    ax_all.axvline(split + 0.5, color="#555555", lw=1, ls="--")
    ax_all.text(split + 8, 0.055, "objective and learning-rate reset", color="#555555", fontsize=9)
    ax_all.set_yscale("log")
    ax_all.set_xlim(1, 700)
    ax_all.set_ylabel("Training loss (log scale)")
    ax_all.set_xlabel("Epoch")
    ax_all.set_title("U3Net on GFS128: complete two-stage training loss")
    ax_all.grid(True, which="both", alpha=.18)
    ax_all.legend(frameon=False, ncol=2, loc="upper right")

    ax_sr.plot(epoch[:split], loss[:split], color=blue, lw=1.25)
    ax_sr.set_xlim(1, 500)
    ax_sr.set_ylim(loss[:split].min() - .002, min(1.62, loss[:split].max() + .003))
    ax_sr.set_xlabel("Epoch")
    ax_sr.set_ylabel(r"Self-recovery loss $L_{sr}$")
    ax_sr.set_title("Stage 1: self-recovery (epochs 1-500)")
    ax_sr.grid(True, alpha=.2)

    ax_sd.plot(epoch[split:], loss[split:], color=orange, lw=1.35, label=r"$L_{sd}$")
    ax_val = ax_sd.twinx()
    ax_val.spines["right"].set_visible(True)
    ax_val.plot(epoch[split:], val[split:], color=green, lw=1, alpha=.55, label="Validation aligned MAE")
    ax_val.scatter(epoch[best], val[best], color=green, s=34, zorder=5)
    ax_val.annotate(f"best validation\nepoch {epoch[best]}: {val[best]:.4f}",
                    (epoch[best], val[best]), xytext=(epoch[best] + 10, val[best] + .055),
                    arrowprops={"arrowstyle": "->", "color": green}, color=green, fontsize=8.5)
    ax_sd.set_xlim(501, 700)
    ax_sd.set_xlabel("Epoch")
    ax_sd.set_ylabel(r"Self-distillation loss $L_{sd}$", color=orange)
    ax_val.set_ylabel("Validation mean-aligned MAE", color=green)
    ax_sd.tick_params(axis="y", colors=orange)
    ax_val.tick_params(axis="y", colors=green)
    ax_sd.set_title("Stage 2: self-distillation (epochs 501-700)")
    ax_sd.grid(True, alpha=.2)

    fig.suptitle("The discontinuity at epoch 501 is caused by switching the loss objective",
                 fontsize=14, weight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "u3net_gfs128_loss_curve.png"
    pdf = OUT / "u3net_gfs128_loss_curve.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
