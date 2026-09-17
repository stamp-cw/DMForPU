"""Compare a current FDU checkpoint, saved deep outputs, and classical baselines."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluate_pretrained_smoke import load_checkpoint, load_config, load_pair  # noqa: E402
from experiments.evaluate_traditional_baselines import metrics  # noqa: E402
from experiments.reporting import environment_metadata, write_markdown_report  # noqa: E402
from diffusion.diffusion_setup import DiffusionSetup  # noqa: E402
from traditional.phase_unwrapping import METHODS  # noqa: E402


SAVED_METHODS = {
    "DLPU_saved": "dlpu",
    "PUNet_saved": "punet",
    "Restormer_saved": "restormer",
    "SqdLstm_saved_pre_fix": "sqd_lstm",
    "U3Net_saved_pre_fix": "u3net",
    "Uformer_saved": "uformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--saved-root", type=Path, default=ROOT / "article" / "data")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def sample_paths(root: Path, count: int) -> list[tuple[str, Path, Path]]:
    inputs = {path.stem: path for path in (root / "test_in").glob("*.mat")}
    targets = {path.stem: path for path in (root / "test_gt").glob("*.mat")}
    keys = sorted(inputs.keys() & targets.keys())[:count]
    if len(keys) < count:
        raise FileNotFoundError(f"requested {count} pairs, found {len(keys)} under {root}")
    return [(key, inputs[key], targets[key]) for key in keys]


def load_saved_predictions(saved_root: Path, sample_index: int) -> dict[str, np.ndarray]:
    batch_file = f"samples_{2 * sample_index}_{2 * sample_index + 1}.pt"
    predictions = {}
    for display_name, directory_name in SAVED_METHODS.items():
        path = saved_root / directory_name / batch_file
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        predictions[display_name] = payload["pred_unwrapped"].detach().cpu().squeeze().numpy()
    return predictions


def align_global_cycle(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    global_k = int(np.rint(np.median((target - prediction) / (2 * np.pi))))
    return prediction + global_k * 2 * np.pi


def save_comparison_figures(
    output: Path,
    wrapped: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    aligned = {name: align_global_cycle(value, target) for name, value in predictions.items()}
    panels = [(wrapped, "Wrapped input", "twilight"), (target, "Ground truth", "turbo")]
    panels.extend((value, name, "turbo") for name, value in aligned.items())
    columns = 4
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.8 * rows), squeeze=False)
    limits = (float(target.min()), float(target.max()))
    for index, (axis, (image, title, cmap)) in enumerate(zip(axes.flat, panels)):
        kwargs = {} if index == 0 else {"vmin": limits[0], "vmax": limits[1]}
        shown = axis.imshow(image, cmap=cmap, **kwargs)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "predictions_first_sample.png", dpi=160)
    plt.close(figure)

    errors = {name: np.abs(value - target) for name, value in aligned.items()}
    error_limit = max(float(np.percentile(np.concatenate([x.ravel() for x in errors.values()]), 99)), 1e-8)
    rows = math.ceil(len(errors) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.8 * rows), squeeze=False)
    for axis, (name, error) in zip(axes.flat, errors.items()):
        shown = axis.imshow(error, cmap="magma", vmin=0, vmax=error_limit)
        axis.set_title(f"{name} |error|", fontsize=9)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    for axis in axes.flat[len(errors):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "errors_first_sample.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.max_samples > 5:
        raise ValueError("max-samples must be between 1 and 5 for the saved article outputs")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = load_config(args.config)
    device = torch.device(config.sampling.device)
    diffusion = DiffusionSetup(config, config.logger).diffusion
    load_checkpoint(diffusion, args.checkpoint, device)
    diffusion.setup_eval()
    pairs = sample_paths(args.data_root, args.max_samples)

    warmup_batch = load_pair(pairs[0][1], pairs[0][2], "synthetic", config)
    warmup_wrapped = warmup_batch["wrapped"].squeeze().numpy()
    for warmup_index in range(args.warmup):
        torch.manual_seed(args.seed - warmup_index - 1)
        torch.cuda.manual_seed_all(args.seed - warmup_index - 1)
        diffusion.setup_data(warmup_batch)
        diffusion.infer_sample()
        for method in METHODS.values():
            method(warmup_wrapped)
    if device.type == "cuda":
        torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    fdu_latencies = []
    first_visual = None
    for sample_index, (key, input_path, target_path) in enumerate(pairs):
        batch = load_pair(input_path, target_path, "synthetic", config)
        wrapped = batch["wrapped"].squeeze().numpy()
        target = batch["unwrapped"].squeeze().numpy()
        predictions = load_saved_predictions(args.saved_root, sample_index)

        sample_fdu_latencies = []
        for _ in range(args.repeats):
            torch.manual_seed(args.seed + sample_index)
            torch.cuda.manual_seed_all(args.seed + sample_index)
            diffusion.setup_data(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            diffusion.infer_sample()
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_fdu_latencies.append((time.perf_counter() - started) * 1000)
        fdu_latency = float(np.mean(sample_fdu_latencies))
        fdu_latencies.append(fdu_latency)
        predictions = {"FDU_current": diffusion.pred_unwrapped.detach().cpu().squeeze().numpy(), **predictions}

        for method_name, method in METHODS.items():
            method_latencies = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                predictions[method_name] = method(wrapped)
                method_latencies.append((time.perf_counter() - started) * 1000)
            latency = float(np.mean(method_latencies))
            result = metrics(predictions[method_name], target, wrapped)
            result.update({"sample": key, "method": method_name, "latency_ms": latency, "source": "rerun"})
            rows.append(result)

        for method_name, prediction in predictions.items():
            if method_name in METHODS:
                continue
            result = metrics(prediction, target, wrapped)
            result.update(
                {
                    "sample": key,
                    "method": method_name,
                    "latency_ms": fdu_latency if method_name == "FDU_current" else None,
                    "source": "rerun" if method_name == "FDU_current" else "saved_output",
                }
            )
            rows.append(result)
        if first_visual is None:
            first_visual = (wrapped, target, predictions)

    method_order = list(dict.fromkeys(row["method"] for row in rows))
    summary = {}
    for method_name in method_order:
        method_rows = [row for row in rows if row["method"] == method_name]
        summary[method_name] = {
            metric_name: float(np.mean([float(row[metric_name]) for row in method_rows]))
            for metric_name in (
                "aligned_mae", "aligned_rmse", "aligned_nrmse", "pge",
                "rewrap_circular_mae", "rewrap_ratio_gt_pi_over_10",
            )
        }
        measured_latencies = [float(row["latency_ms"]) for row in method_rows if row["latency_ms"] is not None]
        summary[method_name]["latency_ms"] = float(np.mean(measured_latencies)) if measured_latencies else None
        summary[method_name]["source"] = method_rows[0]["source"]

    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "environment": environment_metadata(str(device)),
        "protocol": {
            "data_root": str(args.data_root.resolve()),
            "saved_root": str(args.saved_root.resolve()),
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "samples": len(pairs),
            "sample_ids": [key for key, _, _ in pairs],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "summary": summary,
        "per_sample": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(
        args.output / "summary.md",
        "小批量方法效果对比",
        report["environment"],
        report["protocol"],
        (
            (
                "统一指标（样本均值）",
                ("方法", "来源", "对齐MAE", "对齐RMSE", "对齐NRMSE", "PGE", "重缠绕MAE", "延迟/ms"),
                tuple(
                    (
                        name, values["source"], values["aligned_mae"], values["aligned_rmse"],
                        values["aligned_nrmse"], values["pge"], values["rewrap_circular_mae"], values["latency_ms"],
                    )
                    for name, values in sorted(summary.items(), key=lambda item: item[1]["aligned_mae"])
                ),
            ),
        ),
        (
            "FDU_current 与三种传统方法在本次运行中重新推理；其余深度方法来自 article/data 保存输出。",
            "SqdLstm_saved_pre_fix 和 U3Net_saved_pre_fix 是移植修复前历史输出，不能作为最终论文结果。",
            "保存输出无法恢复原始计时环境，因此其延迟记为 N/A；不要与本次重跑延迟混用。",
            "这只是少量样本 smoke comparison，不代表完整测试集结论。",
        ),
    )
    assert first_visual is not None
    save_comparison_figures(args.output, *first_visual)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
