"""Select the HF diffusion inference-step count on GFS128 validation data.

The test sets are not used for selection.  All candidates share the same
checkpoint, validation samples, batching, and random seed.  The strongest
coarse candidates are confirmed with three seeds before the selected setting
and the historical five-step setting are evaluated on every test condition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from diffusion.dcc_wwfca_diffusion import DCCWWFCADiffusion
from experiments import train_gfs_rme128 as protocol

OUT = ROOT / "experiments" / "results" / "hf_inference_steps_gfs128"
CHECKPOINT = ROOT / "experiments" / "results" / "gfs_rme128" / "runs" / "GFS128" / "hf_matched" / "weights" / "epoch_299.pth"
STEPS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 75, 100)
COARSE_SEED = 20000
CONFIRM_SEEDS = (20000, 20001, 20002)
TEST_SEED = 30000
SELECTION_KEY = "u3_aligned_nrmse"


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_indices(path: Path, indices: list[int]):
    indices = np.asarray(indices, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        wrapped = torch.from_numpy(handle["psi"][indices])[:, None]
        target = torch.from_numpy(handle["phi"][indices])[:, None]
        snr = torch.from_numpy(handle["snr"][indices])
    return wrapped, target, snr


def read_split(path: Path):
    with h5py.File(path, "r") as handle:
        wrapped = torch.from_numpy(handle["psi"][:])[:, None]
        target = torch.from_numpy(handle["phi"][:])[:, None]
        snr = torch.from_numpy(handle["snr"][:])
    return wrapped, target, snr


@torch.inference_mode()
def evaluate(model, split, steps: int, batch_size: int, seed: int):
    wrapped, target, snr = split
    sums: dict[str, float] = {}
    count = 0
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, len(wrapped), batch_size):
        wi = wrapped[offset:offset + batch_size].cuda(non_blocking=True)
        ti = target[offset:offset + batch_size].cuda(non_blocking=True)
        si = snr[offset:offset + batch_size].cuda(non_blocking=True)
        std = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., si / 10))
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model.sample(wi, std, generator=generator, steps=steps)
        if prediction.shape != ti.shape or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Invalid prediction at {steps} inference steps")
        for key, value in protocol.metric_sums(prediction, ti, wi).items():
            sums[key] = sums.get(key, 0.0) + value
        count += len(wi)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics = {key: value / count for key, value in sums.items()}
    metrics.update(
        elapsed_seconds=elapsed,
        samples_per_second=count / elapsed,
        milliseconds_per_image=1000 * elapsed / count,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
        samples=count,
    )
    return metrics


def metric_columns(metrics: dict):
    return {key: float(value) for key, value in metrics.items()}


def plot_results(coarse: list[dict], confirmation_summary: list[dict]):
    steps = np.asarray([row["steps"] for row in coarse])
    errors = np.asarray([row[SELECTION_KEY] for row in coarse])
    speed = np.asarray([row["samples_per_second"] for row in coarse])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(steps, errors, "o-", color="#276FBF", linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Inference steps")
    axes[0].set_ylabel("Validation U3-aligned NRMSE")
    axes[0].grid(alpha=.25)
    axes[1].plot(steps, speed, "o-", color="#D1495B", linewidth=1.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Inference steps")
    axes[1].set_ylabel("Throughput (images/s)")
    axes[1].grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_and_efficiency.png", dpi=220)
    plt.close(fig)

    if confirmation_summary:
        labels = [str(row["steps"]) for row in confirmation_summary]
        means = [row[f"mean_{SELECTION_KEY}"] for row in confirmation_summary]
        stds = [row[f"std_{SELECTION_KEY}"] for row in confirmation_summary]
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.errorbar(labels, means, yerr=stds, marker="o", capsize=4, color="#276FBF")
        ax.set_xlabel("Inference steps")
        ax.set_ylabel("Validation U3-aligned NRMSE (mean ± SD, 3 seeds)")
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(OUT / "confirmation_nrmse.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "status.json", {"state": "loading", "updated_at": time.strftime("%F %T")})
    manifest = protocol.prepare_manifest()
    train_path = ROOT / manifest["datasets"]["GFS128"]["train"]
    validation_indices = manifest["validation_indices"][:16] if args.smoke else manifest["validation_indices"]
    validation = read_indices(train_path, validation_indices)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = DCCWWFCADiffusion("hf_matched", phase_low=-14 * math.pi, phase_high=14 * math.pi).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    candidates = (1, 2) if args.smoke else STEPS

    coarse = []
    coarse_by_step = {}
    for index, steps in enumerate(candidates, 1):
        atomic_json(OUT / "status.json", {"state": "coarse_validation", "candidate": steps,
                    "progress": f"{index}/{len(candidates)}", "updated_at": time.strftime("%F %T")})
        metrics = evaluate(model, validation, steps, args.batch, COARSE_SEED)
        row = {"steps": steps, "seed": COARSE_SEED, **metric_columns(metrics)}
        coarse.append(row)
        coarse_by_step[steps] = row
        write_csv(OUT / "coarse_validation.csv", coarse)
        print(f"coarse steps={steps:3d} {SELECTION_KEY}={metrics[SELECTION_KEY]:.8f} "
              f"speed={metrics['samples_per_second']:.2f} images/s", flush=True)

    if args.smoke:
        plot_results(coarse, [])
        atomic_json(OUT / "status.json", {"state": "smoke_complete", "updated_at": time.strftime("%F %T")})
        return

    top = [row["steps"] for row in sorted(coarse, key=lambda row: row[SELECTION_KEY])[:4]]
    confirm_steps = sorted(set(top + [5]))
    confirmation = []
    for steps in confirm_steps:
        confirmation.append(dict(coarse_by_step[steps], phase="confirmation"))
        for seed in CONFIRM_SEEDS[1:]:
            atomic_json(OUT / "status.json", {"state": "confirmation_validation", "candidate": steps,
                        "seed": seed, "updated_at": time.strftime("%F %T")})
            metrics = evaluate(model, validation, steps, args.batch, seed)
            confirmation.append({"steps": steps, "seed": seed, "phase": "confirmation", **metric_columns(metrics)})
            write_csv(OUT / "confirmation_validation.csv", confirmation)

    confirmation_summary = []
    metric_keys = [key for key in coarse[0] if key not in ("steps", "seed", "elapsed_seconds", "samples",
                                                            "samples_per_second", "milliseconds_per_image",
                                                            "peak_gpu_memory_mb")]
    for steps in confirm_steps:
        records = [row for row in confirmation if row["steps"] == steps]
        summary = {"steps": steps}
        for key in metric_keys:
            values = np.asarray([row[key] for row in records], dtype=np.float64)
            summary[f"mean_{key}"] = float(values.mean())
            summary[f"std_{key}"] = float(values.std(ddof=1))
        confirmation_summary.append(summary)
    write_csv(OUT / "confirmation_summary.csv", confirmation_summary)
    best = min(confirmation_summary, key=lambda row: row[f"mean_{SELECTION_KEY}"])["steps"]

    test_rows = []
    test_steps = sorted(set((5, best)))
    conditions = [("clean", ROOT / "data" / "GFS128" / "test_clean.h5")]
    conditions += [(f"{snr}dB", ROOT / "data" / "GFS128" / f"test_{snr}dB.h5")
                   for snr in protocol.TEST_SNRS]
    for condition_index, (condition, path) in enumerate(conditions):
        split = read_split(path)
        for steps in test_steps:
            atomic_json(OUT / "status.json", {"state": "test", "condition": condition, "steps": steps,
                        "updated_at": time.strftime("%F %T")})
            metrics = evaluate(model, split, steps, args.batch, TEST_SEED + condition_index)
            test_rows.append({"condition": condition, "steps": steps, "seed": TEST_SEED + condition_index,
                              **metric_columns(metrics)})
            write_csv(OUT / "full_test_metrics.csv", test_rows)
            print(f"test {condition:>5s} steps={steps:3d} {SELECTION_KEY}={metrics[SELECTION_KEY]:.8f}", flush=True)
        del split

    plot_results(coarse, confirmation_summary)
    best_validation = next(row for row in confirmation_summary if row["steps"] == best)
    baseline_validation = next(row for row in confirmation_summary if row["steps"] == 5)
    summary = {
        "state": "complete",
        "selection_rule": "minimum three-seed mean validation U3-aligned NRMSE",
        "best_inference_steps": best,
        "historical_baseline_steps": 5,
        "best_validation": best_validation,
        "baseline_validation": baseline_validation,
        "candidate_steps": list(STEPS),
        "coarse_seed": COARSE_SEED,
        "confirmation_seeds": list(CONFIRM_SEEDS),
        "validation_samples": len(validation[0]),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_epoch_field": checkpoint.get("epoch"),
        "batch_size": args.batch,
        "test_policy": "Only selected best and historical 5-step settings evaluated after validation selection",
        "updated_at": time.strftime("%F %T"),
    }
    atomic_json(OUT / "summary.json", summary)
    atomic_json(OUT / "status.json", {"state": "complete", "best_inference_steps": best,
                "updated_at": time.strftime("%F %T")})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
