"""Generate the GFS (U3Net MoGR) and RME phase-unwrapping datasets.

This project extension keeps U3Net's 5,000/1,000 sample counts while replacing
the approximate-clean 60 dB training condition with an exactly noise-free one.
Training conditions are {clean,0,5,10,20,30}; each test condition has 1,000
matched scenes.
Scene parameters are shared across requested resolutions; phase is rendered
and re-wrapped separately at every resolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SNRS = np.asarray([np.inf, 0, 5, 10, 20, 30], dtype=np.float32)
TEST_SNRS = (0, 5, 10, 20, 30)
SIZES = (128, 64, 32)
FAMILIES = ("GFS", "RME")
SIGNAL_POWER = 10 ** 0.1  # Identical to U3Net data.py.


def wrap(x: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * x)).astype(np.float32)


def rescale(x: np.ndarray, p: int) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        raise RuntimeError("degenerate generated phase")
    return (((x - lo) / (hi - lo) * 2.0 - 1.0) * (2 * p * math.pi)).astype(np.float32)


def scene_rng(seed: int, family: str, split: str, index: int) -> np.random.Generator:
    family_code = 117 if family == "GFS" else 241
    split_code = 17 if split == "train" else 53
    return np.random.default_rng(np.random.SeedSequence([seed, family_code, split_code, index]))


def gfs_phase(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """MoGR: mixture of Gaussian functions plus a random ramp."""
    c = int(rng.integers(1, 5)); p = int(rng.integers(1, 8))
    slopes = rng.uniform(0, 0.5, 2)
    shift = int(rng.integers(1, 10))
    amplitudes = rng.integers(50, 1000, c)
    mu_x = rng.integers(20, 235, c); mu_y = rng.integers(20, 235, c)
    sigma_x = rng.integers(10, 45, c); sigma_y = rng.integers(10, 45, c)
    # Render on the original 0..255 coordinate system at every output size.
    axis = np.linspace(0, 255, size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    phase = slopes[0] * xx + slopes[1] * yy + shift
    for j in range(c):
        exponent = (xx-mu_x[j])**2/(2*sigma_x[j]**2) + (yy-mu_y[j])**2/(2*sigma_y[j]**2)
        phase += 0.1 * amplitudes[j] * np.exp(-exponent)
    return rescale(phase, p), {"p": p, "components": c}


def rme_phase(size: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Random matrix enlargement from the U3Net paper, Sec. 4.1."""
    s = int(rng.integers(2, 11)); p = int(rng.integers(1, 8))
    distribution = "uniform" if int(rng.integers(0, 2)) == 0 else "normal"
    interpolation = "bilinear" if int(rng.integers(0, 2)) == 0 else "bicubic"
    matrix = rng.uniform(0, 1, (s, s)) if distribution == "uniform" else rng.normal(0, 1, (s, s))
    mode = Image.Resampling.BILINEAR if interpolation == "bilinear" else Image.Resampling.BICUBIC
    enlarged = np.asarray(Image.fromarray(matrix.astype(np.float32), mode="F").resize((size, size), mode),
                          dtype=np.float32)
    return rescale(enlarged, p), {"p": p, "initial_size": s,
                                  "distribution": distribution, "interpolation": interpolation}


def noise_rng(seed: int, family: str, split: str, size: int, snr: float, index: int):
    return np.random.default_rng(np.random.SeedSequence(
        [seed, 311 if family == "GFS" else 449, 71 if split == "train" else 89, size, int(snr), index]))


def render(family: str, size: int, split: str, index: int, snr: float | None, seed: int):
    rng = scene_rng(seed, family, split, index)
    phi, params = gfs_phase(size, rng) if family == "GFS" else rme_phase(size, rng)
    if snr is None or math.isinf(float(snr)):
        return wrap(phi), phi, params
    std = math.sqrt(SIGNAL_POWER / (10 ** (float(snr) / 10)))
    noise = noise_rng(seed, family, split, size, snr, index).normal(0, std, phi.shape).astype(np.float32)
    return wrap(phi + noise), phi, params


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write_file(path: Path, family: str, size: int, split: str, count: int,
               seed: int, fixed_snr: int | None, noise_free: bool = False) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    if tmp.exists(): tmp.unlink()
    # Keep the train SNR assigned to a scene identical across resolutions.
    snr_rng = np.random.default_rng(np.random.SeedSequence([seed, 991, 1 if family == "GFS" else 2]))
    snrs = (np.full(count, np.inf, np.float32) if noise_free else
            snr_rng.choice(TRAIN_SNRS, count, replace=True).astype(np.float32)
            if fixed_snr is None else np.full(count, fixed_snr, np.float32))
    with h5py.File(tmp, "w") as f:
        chunks = (min(32, count), size, size)
        psi = f.create_dataset("psi", (count,size,size), dtype="f4", chunks=chunks, compression="lzf")
        phi = f.create_dataset("phi", (count,size,size), dtype="f4", chunks=chunks, compression="lzf")
        f.create_dataset("snr", data=snrs, dtype="f4")
        f.create_dataset("scene_id", data=np.arange(count,dtype=np.int64))
        f.attrs.update(family=family, source_name="MoGR" if family=="GFS" else "RME",
                       split=split, image_size=size, seed=seed, generator_version="2.0",
                       noise_free=noise_free,
                       contains_noise_free=bool(np.isposinf(snrs).any()),
                       snr_label="clean" if noise_free else
                       ("mixed" if fixed_snr is None else str(fixed_snr)))
        p_values=[]
        for i in range(count):
            sample_snr=None if noise_free or np.isposinf(snrs[i]) else float(snrs[i])
            w,t,params=render(family,size,split,i,sample_snr,seed)
            psi[i],phi[i]=w,t; p_values.append(params["p"])
            if (i+1)%500==0: print(f"{family}{size} {path.name}: {i+1}/{count}",flush=True)
        f.attrs["p_histogram"] = json.dumps({str(p):p_values.count(p) for p in range(1,8)})
    tmp.replace(path)
    with h5py.File(path,"r") as f:
        psi_values = np.asarray(f["psi"][:])
        phi_values = np.asarray(f["phi"][:])
        stored_snrs = np.asarray(f["snr"][:])
        circular = np.angle(np.exp(1j * psi_values[:min(32,count)]))
        clean_mask = np.isposinf(stored_snrs)
        condition_counts = {"clean": int(clean_mask.sum())}
        condition_counts.update({str(v): int(np.sum(stored_snrs == v)) for v in TEST_SNRS})
        clean_wrap_error = (float(np.max(np.abs(
            psi_values[clean_mask] - wrap(phi_values[clean_mask])))) if clean_mask.any() else None)
        audit=dict(samples=count,shape=list(f["psi"].shape),psi_min=float(psi_values.min()),
                   psi_max=float(psi_values.max()),phi_min=float(phi_values.min()),
                   phi_max=float(phi_values.max()),
                   finite=bool(np.isfinite(psi_values).all() and np.isfinite(phi_values).all()),
                   wrapped_range_valid=bool(np.max(np.abs(circular-psi_values[:min(32,count)]))<1e-5),
                   condition_counts=condition_counts,
                   max_clean_wrap_error=clean_wrap_error)
    audit.update(file=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha256(path))
    return audit


def generate(output: Path, train_count: int, test_count: int, seed: int,
             families=FAMILIES, sizes=SIZES) -> None:
    output = output.resolve()
    existing_path = output / "DATASETS_V2.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else {}
    started=time.time(); manifest={"schema_version":1,"generator":"experiments/generate_u3_datasets.py",
        "seed":seed,"families":{},"protocol":{"train_count":train_count,"train_conditions":["clean",0,5,10,20,30],
        "test_count_per_condition":test_count,"test_snrs":list(TEST_SNRS),
        "test_conditions":["clean",*TEST_SNRS],"sizes":list(sizes),
        "noise_power":"10^0.1 / 10^(SNR/10)","GFS_alias":"MoGR"}}
    for family in families:
        manifest["families"][family]={}
        for size in sizes:
            folder=output/f"{family}{size}"; entries={}
            entries["train"]=write_file(folder/"train.h5",family,size,"train",train_count,seed,None)
            entries["test_clean"]=write_file(folder/"test_clean.h5",family,size,"test",test_count,seed,None,noise_free=True)
            for snr in TEST_SNRS:
                entries[f"test_{snr}dB"]=write_file(folder/f"test_{snr}dB.h5",family,size,"test",test_count,seed,snr)
            manifest["families"][family][str(size)]=entries
            atomic=folder/"manifest.json"; atomic.write_text(json.dumps(entries,indent=2,ensure_ascii=False),encoding="utf-8")
    if "RTS" in existing.get("families", {}):
        manifest["families"]["RTS"] = existing["families"]["RTS"]
    manifest["families"]["RBR"] = existing.get("families", {}).get("RBR", {
        "name":"Real-data Based Reconstruction", "status":"reserved; not generated",
        "generated":False, "note":"Reserved for a future LiCSAR-based dataset."})
    manifest["elapsed_seconds"]=time.time()-started
    (output/"DATASETS_V2.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=ROOT/"data")
    ap.add_argument("--train-count",type=int,default=5000); ap.add_argument("--test-count",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=20240913); ap.add_argument("--families",nargs="+",choices=FAMILIES,default=list(FAMILIES))
    ap.add_argument("--sizes",nargs="+",type=int,choices=SIZES,default=list(SIZES)); args=ap.parse_args()
    generate(args.output,args.train_count,args.test_count,args.seed,tuple(args.families),tuple(args.sizes))
