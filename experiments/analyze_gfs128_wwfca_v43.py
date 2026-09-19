"""Analyze validate-then-retrieve WWFCA-v4.3 against matched controls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments" / "results" / "gfs_rme128" / "runs" / "GFS128"
OUT = ROOT / "experiments" / "results" / "gfs128_wwfca_v43"
METHODS = ("wwfca_v41_off", "wwfca_v41_noattn", "wwfca_v41", "wwfca_v43")


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    histories = {method: json.loads((RUNS / method / "history.json").read_text()) for method in METHODS}
    selected = []
    tests = []
    for method, history in histories.items():
        best = min(history, key=lambda row: row["val_u3_aligned_nrmse"])
        selected.append({"method": method, "selected_epoch": best["epoch"],
                         **{key: value for key, value in best.items() if key.startswith("val_")}})
        complete = json.loads((RUNS / method / "complete.json").read_text())
        for condition, metrics in complete["tests"].items():
            tests.append({"method": method, "condition": condition,
                          "selected_epoch": complete["selected_epoch"], **metrics})
    csv_write(OUT / "validation_metrics.csv", selected)
    csv_write(OUT / "test_metrics.csv", tests)

    averages = {}
    for method in METHODS:
        rows = [row for row in tests if row["method"] == method]
        keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "selected_epoch"]
        averages[method] = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    baseline = averages["wwfca_v41_off"]
    directional = averages["wwfca_v41_noattn"]
    full = averages["wwfca_v41"]
    proposed = averages["wwfca_v43"]
    summary = {
        "selection": "lowest validation U3-aligned NRMSE",
        "averages": averages,
        "v43_u3_nrmse_improvement_vs_off_percent":
            100 * (baseline["u3_aligned_nrmse"] - proposed["u3_aligned_nrmse"]) / baseline["u3_aligned_nrmse"],
        "v43_u3_nrmse_improvement_vs_directional_percent":
            100 * (directional["u3_aligned_nrmse"] - proposed["u3_aligned_nrmse"]) / directional["u3_aligned_nrmse"],
        "v43_u3_nrmse_improvement_vs_v41_percent":
            100 * (full["u3_aligned_nrmse"] - proposed["u3_aligned_nrmse"]) / full["u3_aligned_nrmse"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    panels = (
        ("val_u3_aligned_nrmse", "U3-aligned NRMSE"),
        ("val_mean_aligned_nrmse", "Mean-aligned NRMSE"),
        ("val_u3_aligned_ssim", "U3-aligned SSIM"),
        ("gamma", "Residual gamma"),
        ("residual_rms_ratio", "Injected/base RMS"),
        ("seconds", "Epoch seconds"),
    )
    colors = {METHODS[0]: "#377eb8", METHODS[1]: "#4daf4a", METHODS[2]: "#e41a1c", METHODS[3]: "#984ea3"}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, panels):
        for method, history in histories.items():
            axis.plot([row["epoch"] for row in history], [row.get(key, 0) for row in history],
                      label=method, color=colors[method])
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("GFS128 validate-then-retrieve WWFCA-v4.3")
    fig.savefig(OUT / "training_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
