"""Small, no-training evaluation for pretrained phase-unwrapping checkpoints.

The script intentionally uses only a few samples and draws. It validates four
review-driven experiment paths: physical consistency, global 2π alignment,
predictive uncertainty, and inference efficiency.
"""

from __future__ import annotations

import argparse
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
import yaml
from scipy.stats import spearmanr
from torch.utils.flop_counter import FlopCounterMode


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion.diffusion_setup import DiffusionSetup  # noqa: E402
from diffusion.fdu_ddpm_diffusion import FduDDPMDiffusion  # noqa: F401,E402
from model.fdunet.fdunet import FDUNet  # noqa: F401,E402
from model.fdunet.fdunet_v1 import FDUNetV1  # noqa: F401,E402
from utils.util import dict2namespace, wrap_phase  # noqa: E402
from experiments.reporting import environment_metadata, write_markdown_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--input-kind",
        required=True,
        choices=("synthetic", "insar-real"),
    )
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-flops", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        config = dict2namespace(yaml.safe_load(handle))
    config.mode = "sample"
    config.logger = logging.getLogger("pretrained-smoke")
    return config


def paired_paths(root: Path, input_kind: str) -> list[tuple[Path, Path]]:
    if input_kind == "synthetic":
        input_dir, target_dir = root / "test_in", root / "test_gt"
    else:
        input_dir = root / "test_wrapped_real"
        target_dir = root / "test_absolute_real"

    inputs = {path.stem: path for path in input_dir.glob("*.mat")}
    targets = {path.stem: path for path in target_dir.glob("*.mat")}
    keys = sorted(inputs.keys() & targets.keys())
    if not keys:
        raise FileNotFoundError(
            f"No paired MAT files in {input_dir} and {target_dir}"
        )
    return [(inputs[key], targets[key]) for key in keys]


def load_pair(
    input_path: Path,
    target_path: Path,
    input_kind: str,
    config,
) -> dict[str, torch.Tensor]:
    target_key = "gt" if input_kind == "synthetic" else "output"
    wrapped = torch.from_numpy(sio.loadmat(input_path)["input"]).float()[None, None]
    unwrapped = torch.from_numpy(sio.loadmat(target_path)[target_key]).float()[None, None]
    scale = 2 * math.pi * (config.data.k_max - config.data.k_min)
    unwrapped_norm = torch.clamp(
        (unwrapped - config.data.k_min * 2 * math.pi) / scale,
        0,
        1,
    )
    return {
        "wrapped": wrapped,
        "unwrapped": unwrapped,
        "wrapped_neg_norm": torch.clamp(wrapped / math.pi, -1, 1),
        "unwrapped_neg_norm": unwrapped_norm * 2 - 1,
        "wrapped_cond": torch.cat((torch.sin(wrapped), torch.cos(wrapped)), dim=1),
    }


def load_checkpoint(diffusion, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model"]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
    diffusion.model.load_state_dict(state_dict, strict=True)


def scalar_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    wrapped_input: torch.Tensor,
) -> dict[str, float | int]:
    difference = prediction - target
    global_k = int(
        torch.round(torch.median((target - prediction) / (2 * math.pi))).item()
    )
    aligned = prediction + global_k * 2 * math.pi
    aligned_difference = aligned - target

    rewrapped = wrap_phase(prediction)
    circular_residual = torch.atan2(
        torch.sin(rewrapped - wrapped_input),
        torch.cos(rewrapped - wrapped_input),
    )
    reference_residual = torch.atan2(
        torch.sin(wrap_phase(target) - wrapped_input),
        torch.cos(wrap_phase(target) - wrapped_input),
    )
    reference_offset = torch.atan2(
        torch.sin(reference_residual).mean(),
        torch.cos(reference_residual).mean(),
    )
    reference_residual_aligned = torch.atan2(
        torch.sin(reference_residual - reference_offset),
        torch.cos(reference_residual - reference_offset),
    )
    return {
        "mae": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "global_offset_k": global_k,
        "aligned_mae": aligned_difference.abs().mean().item(),
        "aligned_rmse": aligned_difference.square().mean().sqrt().item(),
        "rewrap_circular_mae": circular_residual.abs().mean().item(),
        "rewrap_circular_rmse": circular_residual.square().mean().sqrt().item(),
        "rewrap_ratio_gt_pi_over_10": (
            circular_residual.abs() > math.pi / 10
        ).float().mean().item(),
        "rewrap_ratio_gt_pi_over_4": (
            circular_residual.abs() > math.pi / 4
        ).float().mean().item(),
        "reference_rewrap_circular_mae": reference_residual.abs().mean().item(),
        "reference_rewrap_circular_rmse": reference_residual.square().mean().sqrt().item(),
        "reference_circular_offset_rad": reference_offset.item(),
        "reference_rewrap_offset_aligned_mae": (
            reference_residual_aligned.abs().mean().item()
        ),
    }


def save_first_sample_figures(
    output: Path,
    wrapped: torch.Tensor,
    target: torch.Tensor,
    mean_prediction: torch.Tensor,
    uncertainty: torch.Tensor,
) -> None:
    wrapped_2d = wrapped.squeeze().numpy()
    target_2d = target.squeeze().numpy()
    prediction_2d = mean_prediction.squeeze().numpy()
    uncertainty_2d = uncertainty.squeeze().numpy()
    error_2d = np.abs(prediction_2d - target_2d)
    cycle_2d = np.abs(
        np.arctan2(
            np.sin(np.arctan2(np.sin(prediction_2d), np.cos(prediction_2d)) - wrapped_2d),
            np.cos(np.arctan2(np.sin(prediction_2d), np.cos(prediction_2d)) - wrapped_2d),
        )
    )

    panels = (
        (wrapped_2d, "Wrapped input", "twilight"),
        (target_2d, "Ground truth", "turbo"),
        (prediction_2d, "Predictive mean", "turbo"),
        (error_2d, "Absolute error", "magma"),
        (uncertainty_2d, "Predictive standard deviation", "magma"),
        (cycle_2d, "Circular rewrap error", "magma"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, (image, title, cmap) in zip(axes.flat, panels):
        shown = axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output / "maps.png", dpi=160)
    plt.close(figure)

    horizontal_gradient = np.abs(np.diff(target_2d, axis=1)).mean(axis=1)
    row = int(np.argmax(horizontal_gradient))
    x_axis = np.arange(target_2d.shape[1])
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(x_axis, target_2d[row], label="Ground truth", linewidth=2)
    axis.plot(x_axis, prediction_2d[row], label="Predictive mean", linewidth=1.5)
    axis.set_xlabel("Column")
    axis.set_ylabel("Unwrapped phase")
    axis.set_title(f"High-gradient cross-section at row {row}")
    uncertainty_axis = axis.twinx()
    uncertainty_axis.fill_between(
        x_axis,
        0,
        uncertainty_2d[row],
        alpha=0.22,
        color="tab:red",
        label="Predictive std",
    )
    uncertainty_axis.set_ylabel("Predictive standard deviation")
    handles, labels = axis.get_legend_handles_labels()
    handles_2, labels_2 = uncertainty_axis.get_legend_handles_labels()
    axis.legend(handles + handles_2, labels + labels_2, loc="upper right")
    figure.tight_layout()
    figure.savefig(output / "high_gradient_profile.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.draws < 2:
        raise ValueError("Use at least one sample and two uncertainty draws")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    device = torch.device(config.sampling.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested by the config but is unavailable")

    diffusion = DiffusionSetup(config, config.logger).diffusion
    load_checkpoint(diffusion, args.checkpoint, device)
    diffusion.setup_eval()
    pairs = paired_paths(args.data_root, args.input_kind)[: args.max_samples]

    parameter_count = sum(parameter.numel() for parameter in diffusion.model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in diffusion.model.parameters()
        if parameter.requires_grad
    )

    # One untimed pass removes model initialization and CUDA kernel startup noise.
    warmup_batch = load_pair(*pairs[0], args.input_kind, config)
    diffusion.setup_data(warmup_batch)
    torch.manual_seed(args.seed - 1)
    torch.cuda.manual_seed_all(args.seed - 1)
    diffusion.infer_sample()
    if device.type == "cuda":
        torch.cuda.synchronize()

    inference_flops = None
    flop_count_error = None
    if not args.skip_flops:
        try:
            diffusion.setup_data(warmup_batch)
            torch.manual_seed(args.seed - 2)
            torch.cuda.manual_seed_all(args.seed - 2)
            with FlopCounterMode(display=False) as flop_counter:
                diffusion.infer_sample()
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_flops = int(flop_counter.get_total_flops())
        except (NotImplementedError, RuntimeError) as error:
            flop_count_error = f"{type(error).__name__}: {error}"

    if device.type == "cuda":
        baseline_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    else:
        baseline_memory = 0

    sample_reports = []
    latencies_ms = []
    first_payload = None
    for sample_index, (input_path, target_path) in enumerate(pairs):
        batch = load_pair(input_path, target_path, args.input_kind, config)
        draws = []
        for draw_index in range(args.draws):
            seed = args.seed + sample_index * args.draws + draw_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            diffusion.setup_data(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            diffusion.infer_sample()
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - started) * 1000)
            draws.append(diffusion.pred_unwrapped.detach().cpu())

        draw_tensor = torch.stack(draws)
        mean_prediction = draw_tensor.mean(dim=0)
        uncertainty = draw_tensor.std(dim=0, unbiased=False)
        target = batch["unwrapped"].cpu()
        wrapped = batch["wrapped"].cpu()
        metrics = scalar_metrics(mean_prediction, target, wrapped)
        absolute_error = (mean_prediction - target).abs().flatten().numpy()
        uncertainty_flat = uncertainty.flatten().numpy()
        correlation = spearmanr(uncertainty_flat, absolute_error).statistic
        metrics["uncertainty_error_spearman"] = (
            float(correlation) if np.isfinite(correlation) else None
        )
        metrics["mean_predictive_std"] = uncertainty.mean().item()
        metrics["max_predictive_std"] = uncertainty.max().item()
        metrics["input_file"] = input_path.name
        sample_reports.append(metrics)

        if first_payload is None:
            first_payload = {
                "wrapped": wrapped,
                "target": target,
                "draws": draw_tensor,
                "mean_prediction": mean_prediction,
                "predictive_std": uncertainty,
            }
            save_first_sample_figures(
                args.output,
                wrapped,
                target,
                mean_prediction,
                uncertainty,
            )

    peak_memory = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    aggregate_keys = [
        key
        for key, value in sample_reports[0].items()
        if isinstance(value, (int, float)) and key != "global_offset_k"
    ]
    aggregate = {
        key: float(np.mean([report[key] for report in sample_reports if report[key] is not None]))
        for key in aggregate_keys
        if any(report[key] is not None for report in sample_reports)
    }
    report = {
        "environment": environment_metadata(str(device)),
        "input_kind": args.input_kind,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "sample_count": len(pairs),
        "draws_per_sample": args.draws,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "full_inference_flops": inference_flops,
        "full_inference_gflops": inference_flops / 1e9 if inference_flops is not None else None,
        "flop_count_error": flop_count_error,
        "latency_ms_mean": float(np.mean(latencies_ms)),
        "latency_ms_std": float(np.std(latencies_ms)),
        "latency_ms_all": latencies_ms,
        "cuda_baseline_memory_mb": baseline_memory / 1024**2,
        "cuda_peak_memory_mb": peak_memory / 1024**2,
        "aggregate_metrics": aggregate,
        "samples": sample_reports,
    }
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    latency_mean = report["latency_ms_mean"]
    inference_steps = getattr(config.diffusion, "num_infer_timesteps", None)
    write_markdown_report(
        args.output / "summary.md",
        "预训练扩散模型实验报告",
        report["environment"],
        {
            "input_kind": args.input_kind,
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "sample_count": len(pairs),
            "draws_per_sample": args.draws,
            "inference_steps": inference_steps,
            "seed": args.seed,
        },
        (
            (
                "计算效率",
                ("参数量", "可训练参数量", "完整推理GFLOPs", "延迟均值/ms", "延迟标准差/ms", "吞吐量/images/s", "CUDA峰值显存/MB"),
                ((parameter_count, trainable_parameter_count, report["full_inference_gflops"], latency_mean, report["latency_ms_std"], 1000.0 / latency_mean, report["cuda_peak_memory_mb"]),),
            ),
            (
                "汇总精度、物理一致性与不确定性",
                ("指标", "均值"),
                tuple((key, value) for key, value in aggregate.items()),
            ),
        ),
        (
            "延迟包含完整扩散采样过程，而不是只计算一次去噪网络前向。",
            "当前脚本是小样本流程验证；正式论文需扩大样本数和重复次数。",
            "FLOPs 由 PyTorch 算子计数器统计完整采样；不支持的自定义或第三方算子可能未计入。",
            f"FLOPs 计数异常：{flop_count_error}" if flop_count_error else "FLOPs 计数未报告的算子限制见运行日志。",
        ),
    )
    torch.save(first_payload, args.output / "first_sample.pt")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
