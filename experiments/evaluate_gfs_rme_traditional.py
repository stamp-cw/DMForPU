"""Evaluate LS, quality-guided, and Schofield-DCT baselines on GFS128/RME128."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traditional.phase_unwrapping import (  # noqa: E402
    unwrap_dct_schofield,
    unwrap_least_squares,
    unwrap_quality_guided,
    wrap_phase,
)

OUT = ROOT / "experiments" / "results" / "gfs_rme128" / "traditional_lsqgdct"
SHARDS = OUT / "shards"
FIGURES = OUT / "figures"
DATASETS = ("GFS128", "RME128")
SNRS = (0, 5, 10, 20, 30)
METHODS = {
    "LS": unwrap_least_squares,
    "QG": unwrap_quality_guided,
    "DCT": unwrap_dct_schofield,
}
METHOD_DETAILS = {
    "LS": "Unweighted gradient-domain least squares; Neumann Poisson solution",
    "QG": "Second-difference reliability-guided maximum spanning tree",
    "DCT": "Schofield sin/cos Laplacian with a DCT Neumann solver and integer-cycle projection",
}
TWO_PI = 2.0 * math.pi


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def metrics(prediction: np.ndarray, target: np.ndarray, wrapped: np.ndarray) -> dict[str, float]:
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    raw = prediction - target
    integer_k = np.round(np.median(-raw) / TWO_PI)
    integer = raw + integer_k * TWO_PI
    mean = raw - raw.mean()
    target_range = max(float(np.ptp(target)), 1e-12)
    gradient_error = np.concatenate((np.diff(mean, axis=0).ravel(), np.diff(mean, axis=1).ravel()))
    cycle = wrap_phase(wrap_phase(prediction) - wrapped)
    return {
        "raw_mae": float(np.mean(np.abs(raw))),
        "raw_rmse": float(np.sqrt(np.mean(raw * raw))),
        "integer_aligned_mae": float(np.mean(np.abs(integer))),
        "integer_aligned_rmse": float(np.sqrt(np.mean(integer * integer))),
        "mean_aligned_mae": float(np.mean(np.abs(mean))),
        "mean_aligned_rmse": float(np.sqrt(np.mean(mean * mean))),
        "mean_aligned_nrmse": float(np.sqrt(np.mean(mean * mean)) / target_range),
        "pge": float(np.mean(np.abs(gradient_error))),
        "rewrap_circular_mae": float(np.mean(np.abs(cycle))),
    }


def evaluate_chunk(task: tuple[str, int, str, int, int]) -> list[dict]:
    dataset, snr, method, begin, end = task
    fn = METHODS[method]
    path = ROOT / "data" / dataset / f"test_{snr}dB.h5"
    rows = []
    with h5py.File(path, "r") as f:
        wrapped = f["psi"][begin:end]
        target = f["phi"][begin:end]
        scene_ids = f["scene_id"][begin:end]
    for offset, (wi, ti, scene_id) in enumerate(zip(wrapped, target, scene_ids)):
        wall_start = time.perf_counter(); cpu_start = time.process_time()
        prediction = fn(wi)
        cpu_ms = (time.process_time() - cpu_start) * 1000.0
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        if prediction.shape != ti.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid prediction: {dataset}/{snr}/{method}/{begin+offset}")
        rows.append(
            {
                "dataset": dataset,
                "snr": snr,
                "method": method,
                "sample": begin + offset,
                "scene_id": int(scene_id),
                "algorithm_wall_ms": wall_ms,
                "algorithm_cpu_ms": cpu_ms,
                **metrics(prediction, ti, wi),
            }
        )
    return rows


def evaluate_split(pool: ProcessPoolExecutor, dataset: str, snr: int, method: str, workers: int) -> list[dict]:
    shard = SHARDS / f"{dataset}_{snr}dB_{method}.json"
    if shard.exists():
        return json.loads(shard.read_text(encoding="utf-8"))
    count = 1000
    bounds = np.linspace(0, count, workers + 1, dtype=int)
    tasks = [(dataset, snr, method, int(bounds[i]), int(bounds[i + 1])) for i in range(workers)]
    begin = time.perf_counter()
    rows = [row for part in pool.map(evaluate_chunk, tasks) for row in part]
    rows.sort(key=lambda row: row["sample"])
    elapsed = time.perf_counter() - begin
    if len(rows) != count:
        raise RuntimeError(f"expected {count} rows, got {len(rows)}")
    payload = {"wall_seconds": elapsed, "workers": workers, "rows": rows}
    atomic_json(shard, payload)
    print(f"{dataset} {snr:>2} dB {method}: {elapsed:.2f}s, MAE={np.mean([r['mean_aligned_mae'] for r in rows]):.5f}", flush=True)
    return payload


def load_or_evaluate(pool: ProcessPoolExecutor, dataset: str, snr: int, method: str, workers: int) -> dict:
    shard = SHARDS / f"{dataset}_{snr}dB_{method}.json"
    if shard.exists():
        payload = json.loads(shard.read_text(encoding="utf-8"))
        print(f"reuse {dataset} {snr:>2} dB {method}", flush=True)
        return payload
    return evaluate_split(pool, dataset, snr, method, workers)


def aggregate(all_rows: list[dict], split_wall: dict[tuple[str, int, str], float]) -> list[dict]:
    summary = []
    metric_names = [
        "raw_mae", "raw_rmse", "integer_aligned_mae", "integer_aligned_rmse",
        "mean_aligned_mae", "mean_aligned_rmse", "mean_aligned_nrmse", "pge",
        "rewrap_circular_mae", "algorithm_wall_ms", "algorithm_cpu_ms",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            for snr in SNRS:
                chosen = [r for r in all_rows if r["dataset"] == dataset and r["method"] == method and r["snr"] == snr]
                row = {"dataset": dataset, "method": method, "snr": snr, "samples": len(chosen), "parameters": 0, "training_required": False}
                for key in metric_names:
                    values = np.asarray([r[key] for r in chosen], np.float64)
                    row[key] = float(values.mean())
                    row[key + "_std"] = float(values.std(ddof=1))
                elapsed = split_wall[(dataset, snr, method)]
                row["evaluation_wall_seconds"] = elapsed
                row["evaluation_throughput_images_s"] = len(chosen) / elapsed
                summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def plot_curves(summary: list[dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"LS": "#3178b5", "QG": "#d05f2d", "DCT": "#288a68"}
    for dataset in DATASETS:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        for method in METHODS:
            rows = sorted((r for r in summary if r["dataset"] == dataset and r["method"] == method), key=lambda r: r["snr"])
            x = [r["snr"] for r in rows]
            axes[0].plot(x, [r["mean_aligned_mae"] for r in rows], "o-", color=colors[method], label=method)
            axes[1].plot(x, [100 * r["mean_aligned_nrmse"] for r in rows], "o-", color=colors[method], label=method)
            axes[2].plot(x, [r["pge"] for r in rows], "o-", color=colors[method], label=method)
        axes[0].set(xlabel="SNR (dB)", ylabel="Mean-aligned MAE (rad)")
        axes[1].set(xlabel="SNR (dB)", ylabel="Mean-aligned NRMSE (%)")
        axes[2].set(xlabel="SNR (dB)", ylabel="PGE (rad/pixel)")
        for ax in axes: ax.grid(alpha=.25); ax.legend(frameon=False)
        fig.suptitle(f"{dataset}: traditional phase-unwrapping baselines")
        fig.savefig(FIGURES / f"metrics_{dataset}.png", dpi=190)
        plt.close(fig)


def qualitative() -> None:
    for dataset in DATASETS:
        fig, axes = plt.subplots(3, 5, figsize=(14, 8.5), constrained_layout=True)
        for row, snr in enumerate((0, 10, 30)):
            index = 0
            with h5py.File(ROOT / "data" / dataset / f"test_{snr}dB.h5", "r") as f:
                wrapped, target = f["psi"][index], f["phi"][index]
            predictions = {name: fn(wrapped) for name, fn in METHODS.items()}
            axes[row, 0].imshow(wrapped, cmap="twilight_shifted", vmin=-math.pi, vmax=math.pi)
            axes[row, 1].imshow(target, cmap="turbo")
            axes[row, 0].set_ylabel(f"{snr} dB", fontsize=10, weight="bold")
            if row == 0:
                axes[row, 0].set_title("Wrapped input"); axes[row, 1].set_title("Ground truth")
            errors = []
            for prediction in predictions.values():
                aligned = prediction - np.mean(prediction - target)
                errors.append(np.abs(aligned - target))
            vmax = max(float(np.quantile(error, .99)) for error in errors)
            for col, ((name, _), error) in enumerate(zip(predictions.items(), errors), 2):
                axes[row, col].imshow(error, cmap="magma", vmin=0, vmax=vmax)
                if row == 0: axes[row, col].set_title(f"{name} abs. error")
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{dataset}: mean-aligned qualitative comparison, sample 0")
        fig.savefig(FIGURES / f"qualitative_{dataset}.png", dpi=190)
        plt.close(fig)


def write_report(summary: list[dict], workers: int) -> None:
    lines = [
        "# GFS128 / RME128 传统方法结果", "",
        "测试覆盖每个数据集的 0、5、10、20、30 dB，每档 1,000 张。主表为 mean-aligned MAE（rad）。", "",
        "| 数据集 | 方法 | 0 dB | 5 dB | 10 dB | 20 dB | 30 dB | 平均 CPU ms/张 |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for method in METHODS:
            rows = {r["snr"]: r for r in summary if r["dataset"] == dataset and r["method"] == method}
            cpu = np.mean([rows[s]["algorithm_cpu_ms"] for s in SNRS])
            lines.append(f"| {dataset} | {method} | " + " | ".join(f"{rows[s]['mean_aligned_mae']:.4f}" for s in SNRS) + f" | {cpu:.3f} |")
    lines += [
        "", "## 方法口径", "",
        f"- LS：{METHOD_DETAILS['LS']}。",
        f"- QG：{METHOD_DETAILS['QG']}。",
        f"- DCT：{METHOD_DETAILS['DCT']}。",
        "- LS 的 Poisson 方程也由 DCT 快速求解，但其右端项来自包裹梯度散度；这里的 DCT 指 Schofield 复相位 Laplacian 方法，二者不是重复运行。",
        "- 误差同时保存在原始、整数 2π 对齐和均值对齐三种口径；主表使用均值对齐口径以消除相位参考常数。",
        f"- 测试使用 {workers} 个低优先级 CPU 工作进程；效率表中的 CPU ms/张是算法函数本身的进程 CPU 时间。",
        "", "完整结果见 `summary.csv`、`summary.json` 与 `per_image.csv`，图见 `figures/`。", "",
    ]
    (OUT / "REPORT.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    workers = max(1, args.workers)
    OUT.mkdir(parents=True, exist_ok=True); SHARDS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    protocol = {
        "datasets": list(DATASETS), "snrs_db": list(SNRS), "samples_per_snr": 1000,
        "methods": METHOD_DETAILS, "workers": workers, "python": sys.version,
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "metrics": "same definitions as experiments/train_gfs_rme128.py",
        "phase_reference": "raw, global integer-2pi aligned, and global mean aligned all reported",
    }
    atomic_json(OUT / "protocol.json", protocol)
    all_rows, split_wall = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for dataset in DATASETS:
            for snr in SNRS:
                for method in METHODS:
                    payload = load_or_evaluate(pool, dataset, snr, method, workers)
                    all_rows.extend(payload["rows"])
                    split_wall[(dataset, snr, method)] = float(payload["wall_seconds"])
                    atomic_json(OUT / "status.json", {"state": "running", "completed": len(all_rows), "total": len(DATASETS)*len(SNRS)*len(METHODS)*1000, "dataset": dataset, "snr": snr, "method": method})
    summary = aggregate(all_rows, split_wall)
    write_csv(OUT / "per_image.csv", all_rows); write_csv(OUT / "summary.csv", summary)
    atomic_json(OUT / "summary.json", summary)
    plot_curves(summary); qualitative(); write_report(summary, workers)
    atomic_json(OUT / "status.json", {"state": "complete", "samples": len(all_rows), "summary_rows": len(summary)})
    print(f"complete: {OUT}", flush=True)


if __name__ == "__main__":
    main()
