"""Compare the legacy biased noise formula with the migrated upstream formula."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reporting import environment_metadata, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def wrap_phase(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def measured_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    signal_power = signal.square().mean()
    noise_power = noise.square().mean()
    return float(10 * torch.log10(signal_power / noise_power))


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    wrapped = torch.from_numpy(sio.loadmat(args.input)["input"]).float()
    requested_ratio = 10 ** (args.snr_db / 10)

    # Legacy formula retained only for demonstrating the original migration bug.
    wrapped_norm = (wrapped + math.pi) / (2 * math.pi)
    current_std = torch.sqrt(wrapped_norm.square().mean() / requested_ratio)
    torch.manual_seed(args.seed)
    current_noise = current_std * torch.randn_like(wrapped_norm) * (2 * math.pi) - math.pi

    # Formula now used by SyntheticPUMatNoise, matching Unsupervised-PU.
    corrected_std = wrapped.new_tensor(math.sqrt(10**0.1 / requested_ratio))
    torch.manual_seed(args.seed)
    corrected_raw_noise = corrected_std * torch.randn_like(wrapped)
    torch.manual_seed(args.seed)
    corrected_raw_noise_repeat = corrected_std * torch.randn_like(wrapped)

    current_noisy = wrap_phase(wrapped + current_noise)
    corrected_noisy = wrap_phase(wrapped + corrected_raw_noise)
    corrected_effective_noise = corrected_noisy - wrapped
    corrected_circular_noise = wrap_phase(corrected_effective_noise)
    report = {
        "environment": environment_metadata("CPU"),
        "input": str(args.input),
        "requested_snr_db": args.snr_db,
        "legacy_formula": {
            "noise_mean_rad": float(current_noise.mean()),
            "noise_std_rad": float(current_noise.std()),
            "measured_phase_snr_db": measured_snr_db(wrapped, current_noise),
            "wrapped_output_min": float(current_noisy.min()),
            "wrapped_output_max": float(current_noisy.max()),
        },
        "migrated_upstream_formula": {
            "noise_mean_rad": float(corrected_circular_noise.mean()),
            "noise_std_rad": float(corrected_circular_noise.std()),
            "measured_phase_snr_db": measured_snr_db(wrapped, corrected_circular_noise),
            "wrapped_output_min": float(corrected_noisy.min()),
            "wrapped_output_max": float(corrected_noisy.max()),
            "fixed_seed_reproducible": bool(
                torch.equal(corrected_raw_noise, corrected_raw_noise_repeat)
            ),
        },
    }
    with (args.output / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    write_markdown_report(
        args.output / "summary.md",
        "相位噪声公式审计报告",
        report["environment"],
        {
            "input": str(args.input.resolve()),
            "requested_snr_db": args.snr_db,
            "seed": args.seed,
        },
        (
            (
                "噪声统计",
                ("实现", "均值/rad", "标准差/rad", "实测SNR/dB", "输出最小值", "输出最大值"),
                (
                    ("legacy_formula", *report["legacy_formula"].values()),
                    (
                        "migrated_upstream_formula",
                        report["migrated_upstream_formula"]["noise_mean_rad"],
                        report["migrated_upstream_formula"]["noise_std_rad"],
                        report["migrated_upstream_formula"]["measured_phase_snr_db"],
                        report["migrated_upstream_formula"]["wrapped_output_min"],
                        report["migrated_upstream_formula"]["wrapped_output_max"],
                    ),
                ),
            ),
        ),
        ("该脚本用于公式审计，不包含模型推理，因此不报告参数量、FLOPs 或 GPU 显存。",),
    )

    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    panels = (
        (wrapped, "Original wrapped phase", "twilight"),
        (current_noisy, "Current formula output", "twilight"),
        (corrected_noisy, "Corrected phase-noise output", "twilight"),
        (current_noise, "Current noise", "coolwarm"),
        (corrected_circular_noise, "Migrated circular noise", "coolwarm"),
    )
    for axis, (image, title, cmap) in zip(axes.flat[:5], panels):
        shown = axis.imshow(image.numpy(), cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    axes.flat[5].hist(current_noise.flatten().numpy(), bins=60, alpha=0.6, label="current")
    axes.flat[5].hist(
        corrected_circular_noise.flatten().numpy(), bins=60, alpha=0.6, label="migrated"
    )
    axes.flat[5].set_title("Noise distribution")
    axes.flat[5].legend()
    figure.tight_layout()
    figure.savefig(args.output / "noise_audit.png", dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
