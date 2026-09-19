"""GFS128 study: 200 diffusion steps, x0 vs epsilon, HF vs directional H residual.

The x0 variants train for 300 epochs and epsilon variants for 600 epochs with
paired randomness within each prediction target.
Every epoch is checkpointed.  Validation is performed every 20 epochs with a
prediction-type-specific anchor sampler; inference step count and checkpoint
are jointly refined after training without consulting the test sets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from diffusion.dcc_wwfca_diffusion import DCCWWFCADiffusion
from experiments import train_gfs_rme128 as common

OUT = ROOT / "experiments" / "results" / "gfs128_t200_prediction_study"
RUNS = OUT / "runs"
BATCH = 8
TRAIN_STEPS = 200
SEED = 42
VALIDATE_EVERY = 20
SELECTION_KEY = "u3_aligned_nrmse"
INFERENCE_STEPS = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200)
VARIANTS = {
    "hf_x0": {"architecture": "hf_matched", "prediction_type": "sample", "anchor_steps": 5},
    "directional_x0": {"architecture": "wwfca_v41_noattn", "prediction_type": "sample", "anchor_steps": 5},
    "hf_epsilon": {"architecture": "hf_matched", "prediction_type": "epsilon", "anchor_steps": 25},
    "directional_epsilon": {"architecture": "wwfca_v41_noattn", "prediction_type": "epsilon", "anchor_steps": 25},
    "dwhfa_x0": {"architecture": "dwhfa", "prediction_type": "sample", "anchor_steps": 5},
    "dwhfa_epsilon": {"architecture": "dwhfa", "prediction_type": "epsilon", "anchor_steps": 25},
    "hf_epsilon_loss1": {"architecture": "hf_matched", "prediction_type": "epsilon", "anchor_steps": 25, "loss1": True},
    "directional_epsilon_loss1": {"architecture": "wwfca_v41_noattn", "prediction_type": "epsilon", "anchor_steps": 25, "loss1": True},
    "hf_epsilon_loss2": {"architecture": "hf_matched", "prediction_type": "epsilon", "anchor_steps": 25, "loss2": True},
    "directional_epsilon_loss2": {"architecture": "wwfca_v41_noattn", "prediction_type": "epsilon", "anchor_steps": 25, "loss2": True},
    "dwhfa_epsilon_loss2": {"architecture": "dwhfa", "prediction_type": "epsilon", "anchor_steps": 25, "loss2": True},
    "dwhfa_v2_epsilon": {"architecture": "dwhfa_v2", "prediction_type": "epsilon", "anchor_steps": 25},
    "dwhfa_v2_epsilon_loss2": {"architecture": "dwhfa_v2", "prediction_type": "epsilon", "anchor_steps": 25, "loss2": True},
}
EPOCHS_BY_VARIANT = {
    "hf_x0": 300, "directional_x0": 300,
    "hf_epsilon": 600, "directional_epsilon": 600,
    "hf_epsilon_loss1": 600, "directional_epsilon_loss1": 400,
    "hf_epsilon_loss2": 400, "directional_epsilon_loss2": 400,
    "dwhfa_epsilon_loss2": 400,
    "dwhfa_v2_epsilon": 400, "dwhfa_v2_epsilon_loss2": 400,
    "dwhfa_x0": 300, "dwhfa_epsilon": 300,
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train_val():
    manifest = common.prepare_manifest()
    source = ROOT / manifest["datasets"]["GFS128"]["train"]
    wrapped, target, snr = common.read_h5(source)
    train_indices = torch.tensor(manifest["train_indices"])
    val_indices = torch.tensor(manifest["validation_indices"])
    return ((wrapped[train_indices, None], target[train_indices, None], snr[train_indices]),
            (wrapped[val_indices, None], target[val_indices, None], snr[val_indices]))


def read_test(condition: str):
    name = "test_clean.h5" if condition == "clean" else f"test_{condition}dB.h5"
    wrapped, target, snr = common.read_h5(ROOT / "data" / "GFS128" / name)
    return wrapped[:, None], target[:, None], snr


def build_model(variant: str):
    spec = VARIANTS[variant]
    model = DCCWWFCADiffusion(spec["architecture"], phase_low=-14 * math.pi,
                              phase_high=14 * math.pi)
    model.scheduler = DDPMScheduler(
        num_train_timesteps=TRAIN_STEPS,
        prediction_type=spec["prediction_type"],
        clip_sample=False,
    )
    model.cfg.train_steps = TRAIN_STEPS
    return model.cuda()


def learning_rate(epoch: int, total_epochs: int):
    if epoch < 10:
        return 2e-4 * (epoch + 1) / 10
    progress = (epoch - 9) / (total_epochs - 10)
    return 2e-6 + .5 * (2e-4 - 2e-6) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(model, split, steps: int, seed: int, batch: int = BATCH, timing: bool = False):
    wrapped, target, snr = split
    sums = {}
    count = 0
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if timing:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.eval()
    for offset in range(0, len(wrapped), batch):
        wi = wrapped[offset:offset + batch].cuda(non_blocking=True)
        ti = target[offset:offset + batch].cuda(non_blocking=True)
        si = snr[offset:offset + batch].cuda(non_blocking=True)
        std = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., si / 10))
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = model.sample(wi, std, generator=generator, steps=steps)
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"non-finite prediction at {steps} steps")
        for key, value in common.metric_sums(prediction, ti, wi).items():
            sums[key] = sums.get(key, 0.0) + value
        count += len(wi)
    if timing:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {key: value / count for key, value in sums.items()}
    result.update(samples=count, elapsed_seconds=elapsed,
                  samples_per_second=count / elapsed,
                  milliseconds_per_image=1000 * elapsed / count)
    if timing:
        result["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    return result


def protocol(variant: str):
    spec = VARIANTS[variant]
    manifest_path = common.OUT / "manifest.json"
    return {
        "dataset": "GFS128", "variant": variant,
        "architecture": spec["architecture"], "prediction_type": spec["prediction_type"],
        "diffusion_train_steps": TRAIN_STEPS, "epochs": EPOCHS_BY_VARIANT[variant], "batch": BATCH,
        "optimizer": "AdamW", "weight_decay": 1e-4,
        "schedule": "10-epoch linear warmup to 2e-4, cosine decay to 2e-6",
        "loss": ("PEA: bounded BPE-weighted per-sample epsilon MSE + 0.01*alpha_bar-weighted "
                 "per-sample x0 L1; gamma=5, rho=1, raw (unnormalized) BPE weights" if spec.get("loss2") else
                 "MSE(epsilon_pred,epsilon) + 0.01*L1(x0_pred,x0), with x0_pred reconstructed from "
                 "scheduler.alphas_cumprod" if spec.get("loss1") else
                 "MSE(prediction,target); target=x0 normalized phase for sample, sampled Gaussian noise for epsilon"),
        "lambda_x0": 0.01 if spec.get("loss1") or spec.get("loss2") else 0.0,
        "loss2_parameters": {"bpe_gamma": 5.0, "phase_rho": 1.0,
                              "normalize_bpe_weight": False} if spec.get("loss2") else None,
        "checkpointing": "every epoch plus resumable last.pth",
        "anchor_validation": {"every_epochs": VALIDATE_EVERY,
                              "steps": spec["anchor_steps"], "seed": 20000,
                              "samples": 500, "selection_metric": SELECTION_KEY},
        "post_training_selection": "best fixed-step anchor validation checkpoint; x0=5 steps, epsilon=25 steps",
        "paired_randomness": "same epoch order, timestep seed, diffusion-noise seed across four variants",
        "seed": SEED,
        "manifest_sha256": sha256(manifest_path),
    }


def train(variant: str, smoke_epochs: int | None = None):
    folder = RUNS / variant
    folder.mkdir(parents=True, exist_ok=True)
    # The loss1 HF experiment was originally run for 400 epochs and is now
    # intentionally extended to 600.  Do not let the old completion marker
    # prevent resuming that experiment.
    is_extension = variant == "hf_epsilon_loss1" and EPOCHS_BY_VARIANT[variant] > 400
    if smoke_epochs is None and (folder / "training_complete.json").exists() and not is_extension:
        return
    train_split, val_split = load_train_val()
    seed_all()
    model = build_model(variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    expected_protocol = protocol(variant)
    atomic_json(folder / "protocol.json", expected_protocol)
    last_path = folder / "last.pth"
    history = []
    start_epoch = 0
    if smoke_epochs is None and last_path.exists():
        try:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            state_protocol = state["protocol"]
            # Preserve the original 400-epoch checkpoint and optimizer state;
            # only the requested horizon changed for this extension.
            protocol_compatible = state_protocol == expected_protocol
            if (not protocol_compatible and is_extension
                    and state_protocol.get("variant") == expected_protocol.get("variant")
                    and state_protocol.get("dataset") == expected_protocol.get("dataset")
                    and state_protocol.get("diffusion_train_steps") == expected_protocol.get("diffusion_train_steps")
                    and state_protocol.get("epochs") == 400
                    and expected_protocol.get("epochs") == 600):
                protocol_compatible = True
            if not protocol_compatible:
                raise RuntimeError(f"resume protocol mismatch for {variant}")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            history = state["history"]
            start_epoch = state["epoch"] + 1
        except Exception as exc:
            # A process can be interrupted while atomically writing last.pth.
            # Recover from the latest complete model-only checkpoint and the
            # separately flushed history, then continue with a fresh optimizer.
            history_path = folder / "history.json"
            if not history_path.exists():
                raise RuntimeError(f"cannot resume {variant}: corrupted last.pth and no history.json") from exc
            history = json.loads(history_path.read_text(encoding="utf-8"))
            valid_epochs = [int(row["epoch"]) for row in history]
            if not valid_epochs:
                raise RuntimeError(f"cannot resume {variant}: empty history") from exc
            start_epoch = max(valid_epochs)
            recovery_path = folder / "weights" / f"epoch_{start_epoch:03d}.pth"
            recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
            recovery_protocol = recovery.get("protocol")
            recovery_compatible = recovery_protocol == expected_protocol
            if (not recovery_compatible and is_extension
                    and recovery_protocol.get("variant") == expected_protocol.get("variant")
                    and recovery_protocol.get("dataset") == expected_protocol.get("dataset")
                    and recovery_protocol.get("diffusion_train_steps") == expected_protocol.get("diffusion_train_steps")
                    and recovery_protocol.get("epochs") == 400
                    and expected_protocol.get("epochs") == 600):
                recovery_compatible = True
            if not recovery_compatible:
                raise RuntimeError(f"resume protocol mismatch for recovered {variant}") from exc
            model.load_state_dict(recovery["model"])
            print(f"{variant}: recovered from {recovery_path.name} after unreadable last.pth", flush=True)

    total_epochs = EPOCHS_BY_VARIANT[variant]
    epochs = smoke_epochs or total_epochs
    spec = VARIANTS[variant]
    for epoch in range(start_epoch, epochs):
        lr = learning_rate(epoch, total_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        order = torch.randperm(len(train_split[0]), generator=torch.Generator().manual_seed(SEED + epoch))
        generator = torch.Generator(device="cuda").manual_seed(SEED * 100000 + epoch)
        timesteps = torch.randint(TRAIN_STEPS, (len(order),), generator=generator, device="cuda")
        diffusion_noise = torch.randn(train_split[1].shape, generator=generator, device="cuda")
        model.train()
        total_loss = 0.0
        total_loss_epsilon = 0.0
        total_loss_x0 = 0.0
        total_loss_epsilon_raw = 0.0
        total_loss_bpe = 0.0
        total_loss_phase_raw = 0.0
        total_loss_phase_weighted = 0.0
        total_bpe_weight = 0.0
        total_phase_weight = 0.0
        total_x0_pred_mae = 0.0
        seen = 0
        updates = 0
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for offset in range(0, len(order), BATCH):
            indices = order[offset:offset + BATCH]
            wrapped, target, snr = (tensor[indices].cuda(non_blocking=True) for tensor in train_split)
            sigma = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., snr / 10))
            x0 = model.normalize(target)
            noise = diffusion_noise[indices]
            timestep = timesteps[indices]
            noisy = model.scheduler.add_noise(x0, noise, timestep)
            objective = x0 if spec["prediction_type"] == "sample" else noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction, _, _ = model(noisy, wrapped, timestep, sigma)
                if spec.get("loss2"):
                    # Phase-Equivalent Adaptive (PEA) loss. All reductions are
                    # per sample before applying the timestep weights.
                    alpha_bar = model.scheduler.alphas_cumprod.to(
                        device=noisy.device, dtype=torch.float32)[timestep].clamp_min(1e-8)
                    alpha_bar_4d = alpha_bar.view(-1, 1, 1, 1)
                    sqrt_alpha = alpha_bar_4d.sqrt()
                    sqrt_one_minus = (1.0 - alpha_bar_4d).clamp_min(0.0).sqrt()
                    x0_pred = (noisy.float() - sqrt_one_minus * prediction.float()) / sqrt_alpha
                    epsilon_error = (prediction.float() - noise.float()).square()
                    loss_epsilon_raw_per = epsilon_error.mean(dim=(1, 2, 3))
                    ratio = (1.0 - alpha_bar) / (alpha_bar + 1e-8)
                    bpe_weight = ratio / (1.0 + ratio / 5.0)
                    loss_bpe = (bpe_weight * loss_epsilon_raw_per).mean()
                    phase_error = (x0_pred - x0.float()).abs().mean(dim=(1, 2, 3))
                    phase_weight = alpha_bar
                    loss_phase_weighted = (phase_weight * phase_error).mean()
                    loss = loss_bpe + 0.01 * loss_phase_weighted
                    loss_epsilon = loss_epsilon_raw_per.mean()
                    loss_x0 = phase_error.mean()
                elif spec.get("loss1"):
                    # Reconstruct normalized x0 from epsilon prediction using the
                    # active scheduler's cumulative alpha, without detaching the
                    # prediction so the auxiliary loss backpropagates.
                    alpha_bar = model.scheduler.alphas_cumprod.to(
                        device=noisy.device, dtype=torch.float32)[timestep].view(-1, 1, 1, 1)
                    sqrt_alpha = alpha_bar.clamp_min(1e-8).sqrt()
                    sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
                    x0_pred = (noisy.float() - sqrt_one_minus * prediction.float()) / sqrt_alpha
                    loss_epsilon = F.mse_loss(prediction.float(), noise.float())
                    loss_x0 = F.l1_loss(x0_pred, x0.float())
                    loss = loss_epsilon + 0.01 * loss_x0
                else:
                    loss_epsilon = None
                    loss_x0 = None
                    loss = (prediction - objective).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss {variant} epoch {epoch + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(scaler.get_scale() >= before)
            total_loss += float(loss.detach()) * len(indices)
            if spec.get("loss1"):
                total_loss_epsilon += float(loss_epsilon.detach()) * len(indices)
                total_loss_x0 += float(loss_x0.detach()) * len(indices)
            if spec.get("loss2"):
                total_loss_epsilon_raw += float(loss_epsilon_raw_per.detach().mean()) * len(indices)
                total_loss_bpe += float(loss_bpe.detach()) * len(indices)
                total_loss_phase_raw += float(phase_error.detach().mean()) * len(indices)
                total_loss_phase_weighted += float(loss_phase_weighted.detach()) * len(indices)
                total_bpe_weight += float(bpe_weight.detach().mean()) * len(indices)
                total_phase_weight += float(phase_weight.detach().mean()) * len(indices)
                total_x0_pred_mae += float(phase_error.detach().mean()) * len(indices)
            seen += len(indices)

        should_validate = epoch == 0 or (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == epochs
        validation = (evaluate(model, val_split, spec["anchor_steps"], 20000)
                      if should_validate else {})
        elapsed = time.perf_counter() - started
        diagnostics = model.backbone.diagnostics() if spec["architecture"] in ("wwfca_v41_noattn", "dwhfa", "dwhfa_v2") else {}
        row = {
            "epoch": epoch + 1, "lr": lr, "train_loss": total_loss / seen,
            "prediction_type": spec["prediction_type"], "anchor_steps": spec["anchor_steps"],
            "validated": should_validate,
            **{f"val_{key}": value for key, value in validation.items()},
            "updates": updates, "samples": seen, "seconds": elapsed,
            "samples_per_second": seen / elapsed,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            **diagnostics,
        }
        if spec.get("loss1"):
            row.update({"loss_total": row["train_loss"],
                        "loss_epsilon": total_loss_epsilon / seen,
                        "loss_x0": total_loss_x0 / seen})
        if spec.get("loss2"):
            row.update({"loss_total": row["train_loss"],
                        "loss_epsilon_raw": total_loss_epsilon_raw / seen,
                        "loss_bpe": total_loss_bpe / seen,
                        "loss_phase_raw": total_loss_phase_raw / seen,
                        "loss_phase_weighted": total_loss_phase_weighted / seen,
                        "bpe_weight_mean": total_bpe_weight / seen,
                        "phase_weight_mean": total_phase_weight / seen,
                        "x0_pred_mae": total_x0_pred_mae / seen})
        history.append(row)
        write_csv(folder / "history.csv", history)
        atomic_json(folder / "history.json", history)
        checkpoint = {"epoch": epoch, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                      "protocol": expected_protocol, "metrics": row}
        if smoke_epochs is None:
            atomic_save(folder / "weights" / f"epoch_{epoch + 1:03d}.pth", checkpoint)
            atomic_save(last_path, {**checkpoint, "optimizer": optimizer.state_dict(),
                                    "scaler": scaler.state_dict(), "history": history})
            atomic_json(folder / "status.json", {"state": "training", "epoch": epoch + 1,
                        "epochs": total_epochs, "last_train_loss": row["train_loss"],
                        "last_anchor_nrmse": validation.get(SELECTION_KEY),
                        "updated_at": time.strftime("%F %T")})
        print(f"{variant} {epoch + 1}/{epochs} loss={row['train_loss']:.7f} "
              f"anchor={validation.get(SELECTION_KEY, float('nan')):.7f} {elapsed:.1f}s", flush=True)

    if smoke_epochs is None:
        atomic_json(folder / "training_complete.json", {
            "state": "complete", "variant": variant, "epochs": len(history),
            "total_seconds": sum(row["seconds"] for row in history),
            "updated_at": time.strftime("%F %T"),
        })


def checkpoint_candidates(folder: Path):
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    eligible = [row for row in history if row.get("validated") and row.get(f"val_{SELECTION_KEY}") is not None]
    top = sorted(eligible, key=lambda row: row[f"val_{SELECTION_KEY}"])[:3]
    return [int(row["epoch"]) for row in top]


def load_checkpoint_model(variant: str, epoch: int):
    model = build_model(variant)
    state = torch.load(RUNS / variant / "weights" / f"epoch_{epoch:03d}.pth",
                       map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def sweep_variant(variant: str):
    folder = RUNS / variant
    selection_path = folder / "selection.json"
    if selection_path.exists():
        return
    _, validation = load_train_val()
    epochs = checkpoint_candidates(folder)
    coarse = []
    for epoch in epochs:
        model = load_checkpoint_model(variant, epoch)
        for steps in INFERENCE_STEPS:
            metrics = evaluate(model, validation, steps, 21000, timing=True)
            coarse.append({"variant": variant, "epoch": epoch, "steps": steps, "seed": 21000, **metrics})
            write_csv(folder / "inference_step_coarse.csv", coarse)
            print(f"sweep {variant} epoch={epoch} steps={steps} nrmse={metrics[SELECTION_KEY]:.8f}", flush=True)
        del model
        torch.cuda.empty_cache()

    top_configs = [(row["epoch"], row["steps"])
                   for row in sorted(coarse, key=lambda row: row[SELECTION_KEY])[:3]]
    top_configs = list(dict.fromkeys(top_configs))
    confirmation = []
    for epoch, steps in top_configs:
        base = next(row for row in coarse if row["epoch"] == epoch and row["steps"] == steps)
        confirmation.append(dict(base, phase="confirmation"))
        model = load_checkpoint_model(variant, epoch)
        for seed in (21001, 21002):
            metrics = evaluate(model, validation, steps, seed, timing=True)
            confirmation.append({"variant": variant, "epoch": epoch, "steps": steps,
                                 "seed": seed, "phase": "confirmation", **metrics})
            write_csv(folder / "inference_step_confirmation.csv", confirmation)
        del model
        torch.cuda.empty_cache()

    summaries = []
    for epoch, steps in top_configs:
        rows = [row for row in confirmation if row["epoch"] == epoch and row["steps"] == steps]
        summaries.append({
            "epoch": epoch, "steps": steps,
            f"mean_{SELECTION_KEY}": float(np.mean([row[SELECTION_KEY] for row in rows])),
            f"std_{SELECTION_KEY}": float(np.std([row[SELECTION_KEY] for row in rows], ddof=1)),
            "mean_samples_per_second": float(np.mean([row["samples_per_second"] for row in rows])),
        })
    write_csv(folder / "inference_step_confirmation_summary.csv", summaries)
    best = min(summaries, key=lambda row: row[f"mean_{SELECTION_KEY}"])
    atomic_json(selection_path, {
        "variant": variant, "selection": "minimum three-seed mean validation U3-aligned NRMSE",
        "candidate_checkpoint_epochs": epochs, "candidate_inference_steps": list(INFERENCE_STEPS),
        "selected_epoch": best["epoch"], "selected_inference_steps": best["steps"],
        "validation_mean_u3_aligned_nrmse": best[f"mean_{SELECTION_KEY}"],
        "validation_std_u3_aligned_nrmse": best[f"std_{SELECTION_KEY}"],
        "validation_mean_samples_per_second": best["mean_samples_per_second"],
        "updated_at": time.strftime("%F %T"),
    })


def test_variant(variant: str):
    folder = RUNS / variant
    output = folder / "test_metrics.csv"
    if output.exists():
        return
    selection = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    epoch = selection["selected_epoch"]
    steps = selection["selected_inference_steps"]
    model = load_checkpoint_model(variant, epoch)
    rows = []
    for index, condition in enumerate(("clean", "0", "5", "10", "20", "30")):
        metrics = evaluate(model, read_test(condition), steps, 30000 + index, timing=True)
        rows.append({"variant": variant, "condition": condition, "epoch": epoch,
                     "steps": steps, "seed": 30000 + index, **metrics})
        write_csv(output, rows)
    del model
    torch.cuda.empty_cache()


def select_fixed_variant(variant: str):
    """Select a checkpoint at the user-fixed inference step count."""
    folder = RUNS / variant
    selection_path = folder / "selection.json"
    if selection_path.exists():
        return
    history = json.loads((folder / "history.json").read_text(encoding="utf-8"))
    eligible = [row for row in history if row.get("validated") and
                row.get(f"val_{SELECTION_KEY}") is not None]
    if not eligible:
        raise RuntimeError(f"no fixed-step validation records for {variant}")
    best = min(eligible, key=lambda row: row[f"val_{SELECTION_KEY}"])
    steps = VARIANTS[variant]["anchor_steps"]
    atomic_json(selection_path, {
        "variant": variant,
        "selection": "minimum validation U3-aligned NRMSE at user-fixed inference steps",
        "selected_epoch": int(best["epoch"]),
        "selected_inference_steps": steps,
        "validation_u3_aligned_nrmse": best[f"val_{SELECTION_KEY}"],
        "validation_seed": 20000,
        "inference_step_search": False,
        "updated_at": time.strftime("%F %T"),
    })


def analyze():
    selections = []
    tests = []
    histories = {}
    for variant in VARIANTS:
        selections.append(json.loads((RUNS / variant / "selection.json").read_text(encoding="utf-8")))
        tests.extend(list(csv.DictReader((RUNS / variant / "test_metrics.csv").open(encoding="utf-8-sig"))))
        histories[variant] = json.loads((RUNS / variant / "history.json").read_text(encoding="utf-8"))
    numeric = {"epoch", "steps", "seed", "samples", "elapsed_seconds", "samples_per_second",
               "milliseconds_per_image", "peak_gpu_memory_mb", "raw_mae", "raw_rmse",
               "integer_aligned_mae", "integer_aligned_rmse", "mean_aligned_mae",
               "mean_aligned_rmse", "mean_aligned_nrmse", "pge", "rewrap_circular_mae",
               "u3_aligned_mae", "u3_aligned_rmse", "u3_aligned_nrmse", "u3_aligned_ssim",
               "raw_au", "integer_aligned_au", "range_aligned_au"}
    for row in tests:
        for key in numeric & row.keys():
            row[key] = float(row[key])
    write_csv(OUT / "all_test_metrics.csv", tests)
    averages = {}
    metric_keys = [key for key in tests[0] if key in numeric and key not in
                   {"epoch", "steps", "seed", "samples", "elapsed_seconds", "peak_gpu_memory_mb"}]
    for variant in VARIANTS:
        rows = [row for row in tests if row["variant"] == variant]
        averages[variant] = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    summary = {"protocol": {"train_steps": TRAIN_STEPS, "epochs_by_variant": EPOCHS_BY_VARIANT},
               "selections": selections, "test_averages": averages}
    atomic_json(OUT / "summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for variant, history in histories.items():
        axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label=variant)
        validated = [row for row in history if row.get("validated")]
        axes[1].plot([row["epoch"] for row in validated],
                     [row[f"val_{SELECTION_KEY}"] for row in validated], marker="o", label=variant)
    axes[0].set_yscale("log")
    axes[0].set_title("Training objective")
    axes[0].set_xlabel("Epoch")
    axes[1].set_title("Anchor validation U3-NRMSE")
    axes[1].set_xlabel("Epoch")
    labels = list(VARIANTS)
    axes[2].bar(labels, [averages[v][SELECTION_KEY] for v in labels])
    axes[2].set_title("Six-condition mean test U3-NRMSE")
    axes[2].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=220)
    plt.close(fig)


def sweep_all():
    for variant in VARIANTS:
        sweep_variant(variant)
        test_variant(variant)
    analyze()


def finalize_fixed():
    for variant in VARIANTS:
        select_fixed_variant(variant)
        # Test variants sequentially so latency and throughput remain comparable
        # even though accuracy training used two concurrent GPU processes.
        test_variant(variant)
    analyze()


def smoke():
    smoke_root = OUT / "smoke"
    for variant in VARIANTS:
        # A full epoch is intentionally avoided; verify model/objective/sampler on real data.
        train_split, val_split = load_train_val()
        seed_all()
        model = build_model(variant)
        wrapped, target, snr = (tensor[:2].cuda() for tensor in train_split)
        sigma = torch.sqrt(torch.tensor(10 ** .1, device="cuda") / torch.pow(10., snr / 10))
        noise = torch.randn_like(target)
        timestep = torch.randint(TRAIN_STEPS, (2,), device="cuda")
        x0 = model.normalize(target)
        noisy = model.scheduler.add_noise(x0, noise, timestep)
        objective = x0 if VARIANTS[variant]["prediction_type"] == "sample" else noise
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, _, _ = model(noisy, wrapped, timestep, sigma)
            loss = (prediction - objective).square().mean()
        loss.backward()
        with torch.no_grad():
            sample = model.sample(wrapped, sigma, torch.Generator(device="cuda").manual_seed(9), steps=2)
        result = {"variant": variant, "loss": float(loss), "finite_sample": bool(torch.isfinite(sample).all()),
                  "sample_shape": list(sample.shape), "parameters": sum(p.numel() for p in model.parameters())}
        atomic_json(smoke_root / f"{variant}.json", result)
        print(result, flush=True)
        del model, train_split, val_split
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "sweep", "finalize", "analyze", "smoke"))
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.command == "train":
        if not args.variant:
            parser.error("--variant is required for train")
        train(args.variant)
    elif args.command == "sweep":
        sweep_all()
    elif args.command == "analyze":
        analyze()
    elif args.command == "finalize":
        finalize_fixed()
    else:
        smoke()
