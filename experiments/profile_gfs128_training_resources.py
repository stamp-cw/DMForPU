"""Measure one real optimization step for each official-source GFS128 model.

This probe uses the exact model, batch size, input adapter, loss and optimizer from
train_gfs128_upstream.py.  It does not write checkpoints or alter training state.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch
import torch.nn.functional as F

import train_gfs128_upstream as study


def profile(method: str, train_set) -> dict:
    study.seed_all()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = study.build(method)
    opt = study.optimizer(method, model)
    batch = study.BATCH[method]
    w, target, _ = (x[:batch].cuda() for x in train_set)
    gen = torch.Generator(device="cuda").manual_seed(study.SEED * 100000)
    if method in ("punet", "uformer", "restormer"):
        w, target = study.augment(w, target, gen)

    torch.cuda.synchronize()
    started = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(study.model_input(method, w))
    expected = study.model_input(method, target)
    if method == "dlpu":
        loss = F.l1_loss(out, expected)
    elif method == "punet":
        loss = F.mse_loss(out, expected)
    elif method == "uformer":
        loss = torch.sqrt((out - expected).square() + 1e-6).mean()
    else:
        loss = F.l1_loss(out, expected)
    loss.backward()
    if method == "restormer":
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
    opt.step()
    torch.cuda.synchronize()
    result = {
        "method": method,
        "batch": batch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "step_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "loss": float(loss.detach()),
    }
    del model, opt, w, target, out, expected, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", default=["dlpu", "punet", "uformer", "restormer"])
    args = parser.parse_args()
    train_set, _, _ = study.load_data()
    results = []
    for method in args.methods:
        row = profile(method, train_set)
        results.append(row)
        print(json.dumps(row), flush=True)
    study.dump(study.OUT / "resource_profile.json", results)


if __name__ == "__main__":
    main()
