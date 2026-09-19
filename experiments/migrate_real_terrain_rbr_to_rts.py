"""One-time migration: reserve RBR for real-data reconstruction and rename DEM simulation to RTS."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

TEXT_FILES = (
    "dataset/U3SyntheticH5.py",
    "selector/data_selector.py",
    "experiments/add_clean_test_sets.py",
    "experiments/audit_rts_dataset.py",
    "experiments/create_rts_region_vectors.py",
    "experiments/generate_rts_dataset.py",
    "experiments/make_rts_dataset_atlas.py",
    "tests/test_rts_dataset.py",
    "tests/test_clean_datasets.py",
    "docs/DATASETS_V2.zh-CN.md",
    "docs/DATASET_CONSTRUCTION_SECTION.zh-CN.md",
    "docs/RTS_DATASET.zh-CN.md",
    "configs/rts/dataset.yaml",
    "configs/rts/regions.yaml",
    "configs/rts/simulation.yaml",
    "data/LEGACY_DATASETS_READ_ONLY.md",
    "data/rts_sources/README.md",
    "output/pdf/rts_dataset_atlas.json",
    "experiments/results/rts_generation.out.log",
    "experiments/results/rts_generation.err.log",
)


def replace_name(text: str) -> str:
    return (text.replace("Real-terrain Based Reconstruction", "Real-Terrain Simulation")
                .replace("Real-terrain-based Phase Unwrapping Dataset Atlas", "Real-Terrain Simulation Dataset Atlas")
                .replace("real-terrain based RBR", "real-terrain RTS")
                .replace("RBR", "RTS")
                .replace("rbr", "rts"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, obj) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def transform_json(value):
    if isinstance(value, dict):
        return {replace_name(str(key)): transform_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [transform_json(item) for item in value]
    if isinstance(value, str):
        return replace_name(value)
    return value


def main() -> None:
    for relative in TEXT_FILES:
        path = ROOT / relative
        if path.exists():
            original = path.read_text(encoding="utf-8")
            updated = replace_name(original)
            if updated != original:
                path.write_text(updated, encoding="utf-8")

    for size in (128, 64, 32):
        folder = DATA / f"RTS{size}"
        for path in folder.glob("*.h5"):
            with h5py.File(path, "r+") as dataset:
                dataset.attrs["family"] = "RTS"
                dataset.attrs["source_name"] = "Copernicus GLO-30 real-terrain simulation"
        manifest_path = folder / "manifest.json"
        manifest = transform_json(json.loads(manifest_path.read_text(encoding="utf-8")))
        for entry in manifest.values():
            path = ROOT / entry["file"]
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256(path)
        dump(manifest_path, manifest)

    json_paths = (
        DATA / "RTS_DATASET.json",
        DATA / "RTS_QA_REPORT.json",
        DATA / "RTS_STATISTICS.json",
        DATA / "NOISE_FREE_TEST_SETS.json",
        ROOT / "output/pdf/rts_dataset_atlas.json",
    )
    for path in json_paths:
        if path.exists():
            dump(path, transform_json(json.loads(path.read_text(encoding="utf-8"))))

    global_manifest = json.loads((DATA / "DATASETS_V2.json").read_text(encoding="utf-8"))
    release_snapshot = json.loads((DATA / "RTS_DATASET.json").read_text(encoding="utf-8"))
    previous_rts = global_manifest["families"].get("RTS", {})
    global_manifest["families"]["RTS"] = {
        "status": "generated",
        "generated": True,
        "name": "Real-Terrain Simulation",
        "version": "1.1",
        "source": "Copernicus GLO-30 DEM-driven physical simulation",
        "sizes": {},
        "protocol": previous_rts.get("protocol", {
            "train_count": 5000,
            "validation_count_within_train": 500,
            "test_count_per_condition": 1000,
            "train_snrs": [0, 5, 10, 20, 30, 60],
            "test_snrs": [0, 5, 10, 20, 30],
            "test_conditions": ["clean", 0, 5, 10, 20, 30],
            "spatial_split": True,
            "test_cross_region_fraction": 0.30,
            "noise": "complex-domain, spatial coherence modulated",
        }),
        "qa": release_snapshot.get("qa", {}),
        "elapsed_seconds": release_snapshot.get("elapsed_seconds"),
        "full_audit": release_snapshot.get("full_audit", {}),
    }
    global_manifest["families"]["RBR"] = {
        "name": "Real-data Based Reconstruction",
        "status": "reserved; not generated",
        "generated": False,
        "note": "Reserved for a future LiCSAR-based real-data reconstruction dataset.",
    }
    for size in (128, 64, 32):
        folder_manifest = json.loads((DATA / f"RTS{size}" / "manifest.json").read_text(encoding="utf-8"))
        global_manifest["families"]["RTS"]["sizes"][str(size)] = folder_manifest
    dump(DATA / "DATASETS_V2.json", global_manifest)

    release = json.loads((DATA / "RTS_DATASET.json").read_text(encoding="utf-8"))
    release["name"] = "RTS"
    release["full_name"] = "Real-Terrain Simulation"
    release["generator"] = "experiments/generate_rts_dataset.py"
    release["config_files"] = [replace_name(item) for item in release["config_files"]]
    for size in (128, 64, 32):
        release["sizes"][str(size)] = json.loads((DATA / f"RTS{size}" / "manifest.json").read_text(encoding="utf-8"))
    dump(DATA / "RTS_DATASET.json", release)

    clean = json.loads((DATA / "NOISE_FREE_TEST_SETS.json").read_text(encoding="utf-8"))
    for size in (128, 64, 32):
        clean["families"]["RTS"][str(size)] = json.loads((DATA / f"RTS{size}" / "manifest.json").read_text(encoding="utf-8"))["test_clean"]
    dump(DATA / "NOISE_FREE_TEST_SETS.json", clean)

    smoke = ROOT / "experiments/results/rts_smoke"
    if smoke.exists():
        smoke_folder = smoke / "RTS128"
        for path in smoke_folder.glob("*.h5"):
            with h5py.File(path, "r+") as dataset:
                dataset.attrs["family"] = "RTS"
                dataset.attrs["source_name"] = "Copernicus GLO-30 real-terrain simulation"
        smoke_manifest_path = smoke_folder / "manifest.json"
        smoke_manifest = transform_json(json.loads(smoke_manifest_path.read_text(encoding="utf-8")))
        for entry in smoke_manifest.values():
            path = ROOT / entry["file"]
            entry["bytes"] = path.stat().st_size
            entry["sha256"] = sha256(path)
        dump(smoke_manifest_path, smoke_manifest)
        for name in ("RTS_DATASET.json", "RTS_QA_REPORT.json"):
            path = smoke / name
            dump(path, transform_json(json.loads(path.read_text(encoding="utf-8"))))
    print("RTS migration complete; RBR is reserved for real-data reconstruction.")


if __name__ == "__main__":
    main()
