"""Evaluate training-free traditional phase-unwrapping baselines on MAT pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traditional.phase_unwrapping import METHODS, wrap_phase  # noqa: E402
from experiments.reporting import environment_metadata, write_markdown_report  # noqa: E402


DATA_LAYOUTS = {
    "synthetic": ("test_in", "test_gt", "gt"),
    "insar-simulated": ("test_wrapped", "test_absolute", "output"),
    "insar-real": ("test_wrapped_real", "test_absolute_real", "output"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--input-kind", required=True, choices=tuple(DATA_LAYOUTS))
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--noise-snr-db", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def paired_paths(root: Path, input_kind: str) -> list[tuple[str, Path, Path, str]]:
    input_name, target_name, target_key = DATA_LAYOUTS[input_kind]
    input_dir, target_dir = root / input_name, root / target_name
    inputs = {path.stem: path for path in input_dir.glob("*.mat")}
    targets = {path.stem: path for path in target_dir.glob("*.mat")}
    keys = sorted(inputs.keys() & targets.keys())
    if not keys:
        raise FileNotFoundError(f"No paired MAT files in {input_dir} and {target_dir}")
    return [(key, inputs[key], targets[key], target_key) for key in keys]


def load_pair(input_path: Path, target_path: Path, target_key: str) -> tuple[np.ndarray, np.ndarray]:
    wrapped = np.asarray(sio.loadmat(input_path)["input"], dtype=np.float64).squeeze()
    target = np.asarray(sio.loadmat(target_path)[target_key], dtype=np.float64).squeeze()
    if wrapped.ndim != 2 or target.ndim != 2 or wrapped.shape != target.shape:
        raise ValueError(
            f"Invalid pair {input_path.name}: wrapped {wrapped.shape}, target {target.shape}"
        )
    return wrapped, target


def metrics(prediction: np.ndarray, target: np.ndarray, wrapped: np.ndarray) -> dict[str, float | int]:
    difference = prediction - target
    global_k = int(np.rint(np.median((target - prediction) / (2 * np.pi))))
    aligned = prediction + global_k * 2 * np.pi
    aligned_difference = aligned - target
    target_range = float(np.ptp(target))

    pred_gradient = np.concatenate((np.diff(aligned, axis=0).ravel(), np.diff(aligned, axis=1).ravel()))
    target_gradient = np.concatenate((np.diff(target, axis=0).ravel(), np.diff(target, axis=1).ravel()))
    cycle = wrap_phase(wrap_phase(prediction) - wrap_phase(wrapped))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "global_offset_k": global_k,
        "aligned_mae": float(np.mean(np.abs(aligned_difference))),
        "aligned_rmse": float(np.sqrt(np.mean(aligned_difference**2))),
        "aligned_nrmse": float(np.sqrt(np.mean(aligned_difference**2)) / max(target_range, 1e-12)),
        "pge": float(np.mean(np.abs(pred_gradient - target_gradient))),
        "rewrap_circular_mae": float(np.mean(np.abs(cycle))),
        "rewrap_circular_rmse": float(np.sqrt(np.mean(cycle**2))),
        "rewrap_ratio_gt_pi_over_10": float(np.mean(np.abs(cycle) > np.pi / 10)),
        "rewrap_ratio_gt_pi_over_4": float(np.mean(np.abs(cycle) > np.pi / 4)),
    }


def save_figure(output: Path, wrapped: np.ndarray, target: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    panels = [(wrapped, "Wrapped input", "twilight"), (target, "Ground truth", "turbo")]
    aligned_predictions = {}
    for name, prediction in predictions.items():
        global_k = int(np.rint(np.median((target - prediction) / (2 * np.pi))))
        aligned_predictions[name] = prediction + global_k * 2 * np.pi
    panels.extend(
        (prediction, f"{name} (global 2pi aligned)", "turbo")
        for name, prediction in aligned_predictions.items()
    )
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    target_limits = (float(np.min(target)), float(np.max(target)))
    for panel_index, (axis, (image, title, cmap)) in enumerate(zip(axes.flat, panels)):
        image_limits = {} if panel_index == 0 else {"vmin": target_limits[0], "vmax": target_limits[1]}
        shown = axis.imshow(image, cmap=cmap, **image_limits)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "first_sample.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("max-samples/repeats must be positive and warmup non-negative")
    args.output.mkdir(parents=True, exist_ok=False)
    pairs = paired_paths(args.data_root, args.input_kind)[: args.max_samples]
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    first_predictions: dict[str, np.ndarray] = {}
    first_pair: tuple[np.ndarray, np.ndarray] | None = None

    for sample_index, (key, input_path, target_path, target_key) in enumerate(pairs):
        wrapped, target = load_pair(input_path, target_path, target_key)
        if args.noise_snr_db is not None:
            noise_std = math.sqrt(10**0.1 / 10 ** (args.noise_snr_db / 10))
            wrapped = wrap_phase(wrapped + rng.normal(0.0, noise_std, wrapped.shape))
        if first_pair is None:
            first_pair = (wrapped, target)
        for method_name in args.methods:
            method = METHODS[method_name]
            for _ in range(args.warmup):
                method(wrapped)
            latencies = []
            prediction = None
            for _ in range(args.repeats):
                started = time.perf_counter()
                prediction = method(wrapped)
                latencies.append((time.perf_counter() - started) * 1000)
            assert prediction is not None
            result = metrics(prediction, target, wrapped)
            result.update(
                {
                    "sample": key,
                    "method": method_name,
                    "height": wrapped.shape[0],
                    "width": wrapped.shape[1],
                    "latency_ms_mean": float(np.mean(latencies)),
                    "latency_ms_std": float(np.std(latencies)),
                }
            )
            rows.append(result)
            if sample_index == 0:
                first_predictions[method_name] = prediction

    metric_names = [
        "mae", "rmse", "aligned_mae", "aligned_rmse", "aligned_nrmse", "pge",
        "rewrap_circular_mae", "rewrap_circular_rmse", "rewrap_ratio_gt_pi_over_10",
        "rewrap_ratio_gt_pi_over_4", "latency_ms_mean",
    ]
    summary = {}
    for method_name in args.methods:
        method_rows = [row for row in rows if row["method"] == method_name]
        summary[method_name] = {
            name: {
                "mean": float(np.mean([float(row[name]) for row in method_rows])),
                "std": float(np.std([float(row[name]) for row in method_rows])),
            }
            for name in metric_names
        }

    with (args.output / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "environment": environment_metadata("CPU"),
        "protocol": {
            "data_root": str(args.data_root.resolve()),
            "input_kind": args.input_kind,
            "methods": args.methods,
            "samples": len(pairs),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "noise_snr_db": args.noise_snr_db,
            "seed": args.seed,
            "phase_unit": "radian",
            "reference_alignment": "single global integer multiple of 2*pi",
            "snaphu_note": "SNAPHU/MCF is excluded because this dataset invocation supplies no coherence map.",
        },
        "summary": summary,
        "per_sample": rows,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    efficiency_rows = []
    accuracy_rows = []
    for method_name in args.methods:
        values = summary[method_name]
        latency = values["latency_ms_mean"]["mean"]
        efficiency_rows.append(
            (method_name, 0, "N/A", latency, values["latency_ms_mean"]["std"], 1000.0 / latency, "CPU")
        )
        accuracy_rows.append(
            (
                method_name,
                values["aligned_mae"]["mean"],
                values["aligned_rmse"]["mean"],
                values["aligned_nrmse"]["mean"],
                values["pge"]["mean"],
                values["rewrap_circular_mae"]["mean"],
            )
        )
    write_markdown_report(
        args.output / "summary.md",
        "传统相位解缠实验报告",
        report["environment"],
        report["protocol"],
        (
            (
                "计算效率",
                ("方法", "参数量", "FLOPs", "延迟均值/ms", "延迟标准差/ms", "吞吐量/images/s", "设备"),
                efficiency_rows,
            ),
            (
                "精度与物理一致性",
                ("方法", "对齐MAE", "对齐RMSE", "对齐NRMSE", "PGE", "重缠绕圆周MAE"),
                accuracy_rows,
            ),
        ),
        (
            "传统算法无可训练参数，参数量记为 0。",
            "这些方法包含数据相关的排序、堆操作或频域求解，未用神经网络 FLOPs 定义强行换算，记为 N/A。",
            "延迟为 CPU 端到端单图延迟；正式论文应增加重复次数并固定线程数。",
            report["protocol"]["snaphu_note"],
        ),
    )
    assert first_pair is not None
    save_figure(args.output, *first_pair, first_predictions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
