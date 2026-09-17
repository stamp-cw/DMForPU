"""Full SyntheticPUMat128Big training for PUNet, SQD-LSTM and U3Net.

The runner is deliberately independent of the legacy trainer: it validates and
caches every MAT pair, records one row per epoch, writes one model checkpoint
per epoch, supports exact resume, and evaluates only on a held-out test split.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from utils.util import dict2namespace

OUT = ROOT / "experiments" / "results" / "full_200"
DATA = ROOT / "data" / "SyntheticPUMat128Big"
METHODS = ("punet", "sqd_lstm", "u3net")
CONFIGS = {
    "punet": ROOT / "configs" / "punet_synpu_128_big.yaml",
    "sqd_lstm": ROOT / "configs" / "sqd_lstm_synpu_128_big.yaml",
    "u3net": ROOT / "configs" / "u3net_synpu_128_big.yaml",
}
MODEL_MODULES = {
    "punet": "model.unet.punet",
    "sqd_lstm": "model.lstm.sqd_lstm",
    "u3net": "model.u3net.u3net",
}
SEED = 42
TWO_PI = 2 * math.pi


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def paired_names(input_dir: Path, target_dir: Path) -> list[str]:
    inputs = {p.stem for p in input_dir.glob("*.mat")}
    targets = {p.stem for p in target_dir.glob("*.mat")}
    missing_target = sorted(inputs - targets)
    missing_input = sorted(targets - inputs)
    if missing_target or missing_input:
        raise ValueError(f"Unpaired MAT files: missing targets={missing_target[:5]}, missing inputs={missing_input[:5]}")
    return sorted(inputs)


def read_pair(input_path: Path, target_path: Path) -> tuple[np.ndarray, np.ndarray]:
    input_mat = sio.loadmat(input_path)
    target_mat = sio.loadmat(target_path)
    if "input" not in input_mat or "gt" not in target_mat:
        raise KeyError(f"Expected keys input/gt in {input_path.name}")
    wrapped = np.asarray(input_mat["input"], dtype=np.float32).squeeze()
    target = np.asarray(target_mat["gt"], dtype=np.float32).squeeze()
    if wrapped.shape != (128, 128) or target.shape != wrapped.shape:
        raise ValueError(f"Bad shape in {input_path.name}: input={wrapped.shape}, gt={target.shape}")
    if not np.isfinite(wrapped).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite value in {input_path.name}")
    return wrapped, target


def _build_cache(split: str, names: list[str], input_dir: Path, target_dir: Path) -> dict[str, Any]:
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    input_file, target_file = cache / f"{split}_input.npy", cache / f"{split}_target.npy"
    wrapped_mm = np.lib.format.open_memmap(input_file.with_suffix(".npy.tmp"), mode="w+", dtype=np.float32,
                                           shape=(len(names), 1, 128, 128))
    target_mm = np.lib.format.open_memmap(target_file.with_suffix(".npy.tmp"), mode="w+", dtype=np.float32,
                                          shape=(len(names), 1, 128, 128))
    stats = {"wrapped_min": math.inf, "wrapped_max": -math.inf, "target_min": math.inf,
             "target_max": -math.inf, "cycle_mae_sum": 0.0, "cycle_mae_max": 0.0}
    for index, name in enumerate(names):
        w, t = read_pair(input_dir / f"{name}.mat", target_dir / f"{name}.mat")
        wrapped_mm[index, 0], target_mm[index, 0] = w, t
        cycle_mae = float(np.abs(np.angle(np.exp(1j * (t.astype(np.float64) - w)))).mean())
        stats["wrapped_min"] = min(stats["wrapped_min"], float(w.min()))
        stats["wrapped_max"] = max(stats["wrapped_max"], float(w.max()))
        stats["target_min"] = min(stats["target_min"], float(t.min()))
        stats["target_max"] = max(stats["target_max"], float(t.max()))
        stats["cycle_mae_sum"] += cycle_mae
        stats["cycle_mae_max"] = max(stats["cycle_mae_max"], cycle_mae)
        if (index + 1) % 1000 == 0:
            print(f"audit/cache {split}: {index + 1}/{len(names)}", flush=True)
    wrapped_mm.flush(); target_mm.flush()
    del wrapped_mm, target_mm
    input_file.with_suffix(".npy.tmp").replace(input_file)
    target_file.with_suffix(".npy.tmp").replace(target_file)
    stats["cycle_mae_mean"] = stats.pop("cycle_mae_sum") / len(names)
    stats.update(samples=len(names), shape=[len(names), 1, 128, 128], finite=True,
                 input_cache_sha256=sha256(input_file), target_cache_sha256=sha256(target_file))
    return stats


def prepare() -> dict[str, Any]:
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            for kind in ("input", "target"):
                path = OUT / "cache" / f"{split}_{kind}.npy"
                if not path.exists() or sha256(path) != manifest["audit"][split][f"{kind}_cache_sha256"]:
                    raise RuntimeError(f"Cache missing or changed: {path}")
        return manifest

    train_names = paired_names(DATA / "train_in", DATA / "train_gt")
    official_test = paired_names(DATA / "test_in", DATA / "test_gt")
    if len(train_names) != 20000 or len(official_test) != 2000:
        raise ValueError(f"Unexpected dataset size: train={len(train_names)}, official_test={len(official_test)}")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(official_test))
    val_names = [official_test[i] for i in order[:500]]
    test_names = [official_test[i] for i in order[500:]]
    manifest: dict[str, Any] = {
        "dataset": "SyntheticPUMat128Big", "seed": SEED,
        "policy": "all 20000 official training pairs; seeded disjoint 500/1500 validation/test split of official test",
        "splits": {"train": train_names, "val": val_names, "test": test_names}, "audit": {},
    }
    manifest["audit"]["train"] = _build_cache("train", train_names, DATA / "train_in", DATA / "train_gt")
    manifest["audit"]["val"] = _build_cache("val", val_names, DATA / "test_in", DATA / "test_gt")
    manifest["audit"]["test"] = _build_cache("test", test_names, DATA / "test_in", DATA / "test_gt")
    atomic_json(manifest_path, manifest)
    return manifest


def load_config(method: str, epochs: int) -> dict[str, Any]:
    cfg = yaml.safe_load(CONFIGS[method].read_text(encoding="utf-8"))
    cfg["training"]["brand_new_epochs"] = epochs
    cfg["training"]["batch_size"] = 32
    if method == "u3net":
        start = round(epochs * 500 / 700)
        cfg["training"].update(distill_start_epoch=start, distill_epochs=epochs - start)
    return cfg


def build_model(method: str, cfg: dict[str, Any]) -> torch.nn.Module:
    importlib.import_module(MODEL_MODULES[method])
    from selector.model_selector import _MODELS
    namespace = dict2namespace(cfg)
    model = _MODELS[namespace.model.name](namespace).cuda()
    return model


def cache_tensor(split: str, kind: str) -> torch.Tensor:
    array = np.load(OUT / "cache" / f"{split}_{kind}.npy", mmap_mode="r")
    # 128 GB system RAM is available; a private RAM copy avoids random memmap IO each epoch.
    return torch.from_numpy(np.array(array, copy=True))


def u3_inputs(wrapped: torch.Tensor, snr_db: float = 30.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from model.u3net.u3net import grad_op
    gradient = wrap(grad_op(wrapped))
    cond = torch.full((len(wrapped), 1), math.sqrt(10 ** 0.1 / 10 ** (snr_db / 10)),
                      device=wrapped.device, dtype=wrapped.dtype)
    x_init = torch.ones_like(wrapped)
    a_init = torch.zeros(*wrapped.shape, 2, device=wrapped.device, dtype=wrapped.dtype)
    return gradient, cond, x_init, a_init


def predict(model: torch.nn.Module, method: str, wrapped: torch.Tensor) -> torch.Tensor:
    if method != "u3net":
        return model(wrapped)
    gradient, cond, x_init, a_init = u3_inputs(wrapped)
    return model(gradient, cond, x_init, a_init)[0]


def sqd_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    error = pred - target
    variance = (error.square().mean((1, 2, 3)) - error.mean((1, 2, 3)).square()).mean()
    tv = error[:, :, 1:, :].sub(error[:, :, :-1, :]).abs().mean()
    tv = tv + error[:, :, :, 1:].sub(error[:, :, :, :-1]).abs().mean()
    loss = variance + 0.1 * tv
    return loss, {"variance": float(variance.detach()), "tv": float(tv.detach())}


def u3_loss(model: torch.nn.Module, teacher: torch.nn.Module | None, wrapped: torch.Tensor,
            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    from model.u3net.u3net import grad_op
    std = math.sqrt(10 ** 0.1 / 10 ** 3)
    noisy = wrap(wrapped + torch.randn(wrapped.shape, generator=generator, device=wrapped.device) * std)
    plus_grad, cond, x_init, a_init = u3_inputs(noisy)
    if teacher is None:
        pred, stages = model(plus_grad, cond, x_init, a_init)
        minus_grad = grad_op(2 * wrapped - noisy)
        terms = [wrap(minus_grad - grad_op(stage.float())).square().mean() / (len(stages) - i)
                 for i, stage in enumerate(stages)]
        loss = sum(terms)
        return loss, pred, {"self_recovery": float(loss.detach()), "distillation": 0.0}
    with torch.no_grad():
        teacher_target = teacher(plus_grad, cond, x_init, a_init)[0]
    clean_grad, clean_cond, clean_x, clean_a = u3_inputs(wrapped)
    pred, stages = model(clean_grad, clean_cond, clean_x, clean_a)
    target_gradient = grad_op(teacher_target.float())
    terms = [F.l1_loss(grad_op(stage.float()), target_gradient) / (len(stages) - i)
             for i, stage in enumerate(stages)]
    loss = sum(terms)
    return loss, pred, {"self_recovery": 0.0, "distillation": float(loss.detach())}


def metric_sums(pred: torch.Tensor, target: torch.Tensor, wrapped: torch.Tensor) -> dict[str, float]:
    pred, target, wrapped = pred.float(), target.float(), wrapped.float()
    difference = pred - target
    k = torch.round(torch.median((target - pred).flatten(1), dim=1).values / TWO_PI)[:, None, None, None]
    aligned_error = pred + k * TWO_PI - target
    target_range = target.flatten(1).amax(1) - target.flatten(1).amin(1)
    pge = torch.cat((aligned_error[:, :, 1:, :].flatten(1), aligned_error[:, :, :, 1:].flatten(1)), 1).abs().mean(1)
    cycle = wrap(wrap(pred) - wrap(wrapped))
    values = {
        "mae": difference.abs().flatten(1).mean(1),
        "rmse": difference.square().flatten(1).mean(1).sqrt(),
        "aligned_mae": aligned_error.abs().flatten(1).mean(1),
        "aligned_rmse": aligned_error.square().flatten(1).mean(1).sqrt(),
        "aligned_nrmse": aligned_error.square().flatten(1).mean(1).sqrt() / target_range.clamp_min(1e-12),
        "pge": pge,
        "rewrap_circular_mae": cycle.abs().flatten(1).mean(1),
        "rewrap_circular_rmse": cycle.square().flatten(1).mean(1).sqrt(),
    }
    return {key: float(value.sum().cpu()) for key, value in values.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, method: str, wrapped_cpu: torch.Tensor, target_cpu: torch.Tensor,
             batch_size: int) -> dict[str, float]:
    model.eval(); sums: dict[str, float] = {}; count = 0
    for offset in range(0, len(wrapped_cpu), batch_size):
        w = wrapped_cpu[offset:offset + batch_size].cuda()
        t = target_cpu[offset:offset + batch_size].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = predict(model, method, w)
        if p.shape != t.shape or not torch.isfinite(p).all():
            raise RuntimeError(f"Invalid {method} prediction shape/values")
        batch = metric_sums(p, t, w)
        for key, value in batch.items(): sums[key] = sums.get(key, 0.0) + value
        count += len(w)
    return {key: value / count for key, value in sums.items()}


def write_history(folder: Path, history: list[dict[str, Any]]) -> None:
    atomic_json(folder / "history.json", history)
    temp = folder / "history.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    temp.replace(folder / "history.csv")


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train(method: str, epochs: int = 200, batch_size: int = 32) -> None:
    manifest = prepare(); folder = OUT / "runs" / method
    if (folder / "complete.json").exists():
        print(f"{method}: already complete", flush=True); return
    folder.mkdir(parents=True, exist_ok=True)
    cfg = load_config(method, epochs)
    protocol = {"method": method, "epochs": epochs, "batch_size": batch_size, "seed": SEED,
                "config": cfg, "manifest_sha256": sha256(OUT / "manifest.json"),
                "precision": "CUDA AMP bfloat16; unscaled optimizer; losses and metrics accumulated in float32",
                "checkpoint_policy": "model-only every epoch plus resumable last and best"}
    atomic_json(folder / "protocol.json", protocol)
    seed_all(SEED); model = build_model(method, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]),
                                 weight_decay=float(cfg["optim"]["weight_decay"]), eps=float(cfg["optim"]["eps"]))
    # BF16 retains FP32's exponent range; scaling is unnecessary and caused FP16
    # overflow in the audited PUNet pilot at epoch 4.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    teacher = None; start_epoch = 0; best = math.inf; history: list[dict[str, Any]] = []
    last = folder / "last.pth"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["protocol"] != protocol: raise ValueError("Resume protocol differs from current protocol")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"]); start_epoch = state["epoch"] + 1
        best, history = state["best"], state["history"]
        if state.get("teacher") is not None:
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            teacher.load_state_dict(state["teacher"])
    train_w, train_t = cache_tensor("train", "input"), cache_tensor("train", "target")
    val_w, val_t = cache_tensor("val", "input"), cache_tensor("val", "target")
    parameter_count = sum(p.numel() for p in model.parameters())
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        phase_start = cfg["training"].get("distill_start_epoch", epochs + 1)
        if method == "u3net" and epoch >= phase_start and teacher is None:
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]),
                                         weight_decay=float(cfg["optim"]["weight_decay"]), eps=float(cfg["optim"]["eps"]))
            scaler = torch.amp.GradScaler("cuda", enabled=False)
        relative_epoch = epoch - phase_start if teacher is not None else epoch
        lr = float(cfg["optim"]["lr"])
        if method == "u3net": lr *= float(cfg["optim"].get("gamma", 0.99)) ** relative_epoch
        elif epoch < int(cfg["optim"].get("warmup", 0)):
            lr *= (epoch + 1) / int(cfg["optim"]["warmup"])
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device="cuda").manual_seed(SEED * 100000 + epoch)
        permutation = torch.randperm(len(train_w), generator=torch.Generator().manual_seed(SEED + epoch))
        totals = {"loss": 0.0, "mae": 0.0, "variance": 0.0, "tv": 0.0,
                  "self_recovery": 0.0, "distillation": 0.0}
        updates = 0; seen = 0; epoch_start = time.perf_counter()
        for offset in range(0, len(train_w), batch_size):
            indices = permutation[offset:offset + batch_size]
            w, t = train_w[indices].cuda(), train_t[indices].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if method == "u3net": loss, p, parts = u3_loss(model, teacher, w, generator)
                else:
                    p = model(w)
                    if method == "punet": loss, parts = F.l1_loss(p, t), {}
                    else: loss, parts = sqd_loss(p, t)
            if p.shape != t.shape or not torch.isfinite(loss) or not torch.isfinite(p).all():
                raise RuntimeError(f"Non-finite or bad output: {method}, epoch {epoch + 1}, offset {offset}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            if method != "u3net":
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optim"]["grad_clip"]))
                if not torch.isfinite(grad_norm): raise RuntimeError("Non-finite gradient")
            before = scaler.get_scale(); scaler.step(optimizer); scaler.update()
            updates += int(scaler.get_scale() >= before)
            n = len(w); seen += n
            totals["loss"] += float(loss.detach()) * n
            totals["mae"] += float((p.float() - t).abs().mean().detach()) * n
            for key, value in parts.items(): totals[key] += value * n
        torch.cuda.synchronize(); train_seconds = time.perf_counter() - epoch_start
        validation = evaluate(model, method, val_w, val_t, batch_size)
        epoch_seconds = time.perf_counter() - epoch_start
        row: dict[str, Any] = {
            "epoch": epoch + 1, "phase": "distillation" if teacher is not None else "self_recovery" if method == "u3net" else "supervised",
            "lr": lr, "train_loss": totals["loss"] / seen, "train_output_mae": totals["mae"] / seen,
            "train_variance": totals["variance"] / seen, "train_tv": totals["tv"] / seen,
            "train_self_recovery": totals["self_recovery"] / seen,
            "train_distillation": totals["distillation"] / seen,
            **{f"val_{key}": value for key, value in validation.items()},
            "successful_updates": updates, "batches": math.ceil(seen / batch_size), "samples": seen,
            "train_seconds": train_seconds, "epoch_seconds": epoch_seconds,
            "train_samples_per_second": seen / train_seconds,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
        }
        if updates != row["batches"] or not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise RuntimeError(f"Epoch integrity check failed: {row}")
        history.append(row); write_history(folder, history)
        state_dict = cpu_state(model)
        epoch_state = {"method": method, "epoch": epoch + 1, "model": state_dict,
                       "protocol_sha256": sha256(folder / "protocol.json"), "val_aligned_mae": validation["aligned_mae"]}
        atomic_torch(folder / "weights" / f"epoch_{epoch + 1:03d}.pth", epoch_state)
        if validation["aligned_mae"] < best:
            best = validation["aligned_mae"]
            atomic_torch(folder / "best.pth", epoch_state)
        resume = {"protocol": protocol, "epoch": epoch, "model": state_dict, "optimizer": optimizer.state_dict(),
                  "scaler": scaler.state_dict(), "teacher": cpu_state(teacher) if teacher is not None else None,
                  "best": best, "history": history}
        atomic_torch(last, resume)
        atomic_json(folder / "status.json", {"state": "training", "epoch": epoch + 1, "epochs": epochs,
                                             "best_val_aligned_mae": best, "updated": time.strftime("%F %T")})
        print(f"{method} epoch {epoch + 1:03d}/{epochs} loss={row['train_loss']:.6f} "
              f"val_aligned_mae={validation['aligned_mae']:.6f} time={epoch_seconds:.1f}s", flush=True)
    best_state = torch.load(folder / "best.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"])
    test_w, test_t = cache_tensor("test", "input"), cache_tensor("test", "target")
    test_metrics = evaluate(model, method, test_w, test_t, batch_size)
    efficiency = benchmark(model, method, parameter_count)
    result = {"method": method, "selected_epoch": best_state["epoch"], "selection_metric": "val_aligned_mae",
              "best_val_aligned_mae": best, "test_samples": len(test_w), "test": test_metrics,
              "efficiency": efficiency, "wall_seconds_this_invocation": time.perf_counter() - started}
    atomic_json(folder / "complete.json", result)
    atomic_json(folder / "status.json", {"state": "complete", **result})


@torch.no_grad()
def benchmark(model: torch.nn.Module, method: str, parameter_count: int) -> dict[str, Any]:
    model.eval(); sample = cache_tensor("test", "input")[:1].cuda()
    for _ in range(20): predict(model, method, sample)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); times = []
    for _ in range(100):
        begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        begin.record(); predict(model, method, sample); end.record(); end.synchronize()
        times.append(begin.elapsed_time(end))
    state_bytes = sum(v.numel() * v.element_size() for v in model.state_dict().values())
    result = {"parameters": parameter_count, "model_state_mib": state_bytes / 2 ** 20,
              "latency_batch1_ms_mean": float(np.mean(times)), "latency_batch1_ms_median": float(np.median(times)),
              "latency_batch1_ms_p95": float(np.percentile(times, 95)),
              "inference_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
              "inference_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20}
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as counter: predict(model, method, sample)
        result["flops_batch1"] = int(counter.get_total_flops())
    except Exception as error:
        result["flops_error"] = repr(error)
    return result


def smoke(batch_size: int = 2) -> None:
    prepare(); report = {}
    w, t = cache_tensor("train", "input")[:batch_size].cuda(), cache_tensor("train", "target")[:batch_size].cuda()
    for method in METHODS:
        seed_all(SEED); cfg = load_config(method, 200); model = build_model(method, cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"]["lr"]))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if method == "u3net":
                loss, p, _ = u3_loss(model, None, w, torch.Generator(device="cuda").manual_seed(SEED))
            elif method == "sqd_lstm": p = model(w); loss, _ = sqd_loss(p, t)
            else: p = model(w); loss = F.l1_loss(p, t)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        finite_gradient = bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
        optimizer.step()
        metrics = evaluate(model, method, w.cpu(), t.cpu(), batch_size)
        report[method] = {"output_shape": list(p.shape), "loss": float(loss.detach()),
                          "finite_output": bool(torch.isfinite(p).all()), "finite_gradient": finite_gradient,
                          "parameters": sum(x.numel() for x in model.parameters()), "post_step_metrics": metrics}
        del model, optimizer; torch.cuda.empty_cache()
    if not all(r["finite_output"] and r["finite_gradient"] and r["output_shape"] == [batch_size, 1, 128, 128]
               for r in report.values()): raise RuntimeError(report)
    atomic_json(OUT / "smoke_test.json", report); print(json.dumps(report, indent=2), flush=True)


def queue(epochs: int, batch_size: int) -> None:
    prepare(); smoke(2)
    jobs = [["train", "--method", method, "--epochs", str(epochs), "--batch-size", str(batch_size)] for method in METHODS]
    atomic_json(OUT / "queue.json", jobs)
    for index, job in enumerate(jobs):
        atomic_json(OUT / "queue_status.json", {"state": "running", "index": index, "total": len(jobs), "job": job})
        with (OUT / "run.log").open("a", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "-B", "-u", str(Path(__file__).resolve()), *job],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            atomic_json(OUT / "queue_status.json", {"state": "failed", "index": index, "job": job, "exit_code": code})
            raise RuntimeError(f"Job failed: {job}, exit={code}")
    atomic_json(OUT / "queue_status.json", {"state": "complete", "jobs": len(jobs), "updated": time.strftime("%F %T")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "train", "queue"))
    parser.add_argument("--method", choices=METHODS, default="punet")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    if args.command == "prepare": prepare()
    elif args.command == "smoke": smoke(args.batch_size)
    elif args.command == "train": train(args.method, args.epochs, args.batch_size)
    else: queue(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
