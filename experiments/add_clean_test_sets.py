"""Add paired noise-free test sets to the generated GFS, RME, and RTS datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FAMILIES = ("GFS", "RME", "RTS")
SIZES = (128, 64, 32)


def dump(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_clean_file(family: str, size: int, force: bool) -> dict:
    folder = DATA / f"{family}{size}"
    source = folder / "test_30dB.h5"
    target = folder / "test_clean.h5"
    temporary = folder / "test_clean.h5.tmp"
    if target.exists() and not force:
        print(f"reuse {target.relative_to(ROOT)}", flush=True)
    else:
        if temporary.exists():
            temporary.unlink()
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r+") as dataset:
            count = len(dataset["phi"])
            for begin in range(0, count, 32):
                end = min(count, begin + 32)
                if family == "RTS":
                    clean = dataset["phi_wrapped_clean"][begin:end]
                    dataset["noise_map"][begin:end] = np.zeros_like(clean, dtype=np.float16)
                else:
                    phase = dataset["phi"][begin:end]
                    clean = np.angle(np.exp(1j * phase)).astype(np.float32)
                dataset["psi"][begin:end] = clean
            dataset["snr"][:] = np.inf
            dataset.attrs["generator_version"] = "1.1"
            dataset.attrs["noise_free"] = True
            dataset.attrs["noise_model"] = "none"
            dataset.attrs["snr_label"] = "clean"
            dataset.attrs["derived_from"] = source.name
        temporary.replace(target)
        print(f"created {target.relative_to(ROOT)}", flush=True)

    with h5py.File(source, "r") as noisy, h5py.File(target, "r") as clean_file:
        count = len(clean_file["phi"])
        max_error = 0.0
        max_noise_map = 0.0
        psi_min, psi_max = np.inf, -np.inf
        phi_min, phi_max = np.inf, -np.inf
        for begin in range(0, count, 32):
            end = min(count, begin + 32)
            phase = clean_file["phi"][begin:end]
            expected = (clean_file["phi_wrapped_clean"][begin:end] if family == "RTS"
                        else np.angle(np.exp(1j * phase)).astype(np.float32))
            observed = clean_file["psi"][begin:end]
            max_error = max(max_error, float(np.max(np.abs(observed - expected))))
            psi_min = min(psi_min, float(observed.min()))
            psi_max = max(psi_max, float(observed.max()))
            phi_min = min(phi_min, float(phase.min()))
            phi_max = max(phi_max, float(phase.max()))
            if family == "RTS":
                max_noise_map = max(max_noise_map, float(np.max(np.abs(clean_file["noise_map"][begin:end]))))
        if not np.isposinf(clean_file["snr"][:]).all():
            raise RuntimeError(f"{target}: clean SNR is not +inf")
        if max_error != 0 or max_noise_map != 0:
            raise RuntimeError(f"{target}: clean consistency failed ({max_error=}, {max_noise_map=})")
        for key in ("phi", "scene_id"):
            if not np.array_equal(noisy[key][:], clean_file[key][:]):
                raise RuntimeError(f"{target}: paired field {key} changed")
        if family == "RTS":
            for key in ("phi_wrapped_clean", "coherence", "dem", "deformation_phase", "random_seed",
                        "region_id", "partition", "difficulty", "deformation_type", "scene_mode",
                        "satellite_id", "source_row", "source_col"):
                if not np.array_equal(noisy[key][:], clean_file[key][:]):
                    raise RuntimeError(f"{target}: paired field {key} changed")

    result = {
        "samples": count,
        "shape": [count, size, size],
        "psi_min": psi_min,
        "psi_max": psi_max,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "finite": True,
        "wrapped_range_valid": psi_min >= -np.pi - 1e-6 and psi_max < np.pi + 1e-6,
        "noise_free": True,
        "snr_label": "clean",
        "snr_storage": "+inf",
        "max_clean_wrap_error": max_error,
        "file": str(target.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
    }
    if family == "RTS":
        result["max_abs_noise_map"] = max_noise_map
    return result


def update_manifests(entries: dict) -> None:
    for family in FAMILIES:
        for size in SIZES:
            path = DATA / f"{family}{size}" / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["test_clean"] = entries[family][str(size)]
            dump(path, manifest)

    path = DATA / "DATASETS_V2.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = manifest.setdefault("protocol", {})
    protocol["test_count_per_condition"] = protocol.get("test_count_per_snr", 1000)
    protocol["test_conditions"] = ["clean", *protocol.get("test_snrs", [0, 5, 10, 20, 30])]
    for family in ("GFS", "RME"):
        for size in SIZES:
            manifest["families"][family][str(size)]["test_clean"] = entries[family][str(size)]
    rts = manifest["families"]["RTS"]
    rts["version"] = "1.1"
    rts_protocol = rts.setdefault("protocol", {})
    rts_protocol["test_count_per_condition"] = rts_protocol.get("test_count_per_snr", 1000)
    rts_protocol["test_conditions"] = ["clean", *rts_protocol.get("test_snrs", [0, 5, 10, 20, 30])]
    for size in SIZES:
        rts["sizes"][str(size)]["test_clean"] = entries["RTS"][str(size)]
    dump(path, manifest)

    path = DATA / "RTS_DATASET.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["test_conditions"] = ["clean", 0, 5, 10, 20, 30]
    for size in SIZES:
        manifest["sizes"][str(size)]["test_clean"] = entries["RTS"][str(size)]
    dump(path, manifest)

    report = {
        "schema_version": 1,
        "definition": "psi = wrap(phi), with no observation noise",
        "snr_storage": "+inf in the HDF5 snr field",
        "paired_source": "test_30dB.h5 (same phi, scene_id, and scene metadata)",
        "families": entries,
        "state": "pass",
    }
    dump(DATA / "NOISE_FREE_TEST_SETS.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    entries = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        for size in SIZES:
            entries[family][str(size)] = create_clean_file(family, size, args.force)
    update_manifests(entries)
    print("All noise-free test sets passed exact paired-data validation.", flush=True)


if __name__ == "__main__":
    main()
