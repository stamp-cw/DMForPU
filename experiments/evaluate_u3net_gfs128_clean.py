"""Evaluate completed upstream U3Net checkpoints on paired noise-free GFS128."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch

import train_gfs128_upstream as study


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/results/gfs128_upstream/runs/u3net"


def main() -> None:
    with h5py.File(ROOT / "data/GFS128/test_clean.h5", "r") as dataset:
        split = (
            torch.from_numpy(dataset["psi"][:])[:, None],
            torch.from_numpy(dataset["phi"][:])[:, None],
            torch.from_numpy(dataset["snr"][:]),
        )
    result = {}
    for name, path in (("final_epoch_700", RUN / "last.pth"), ("best_validation_epoch_626", RUN / "best.pth")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = study.build("u3net")
        model.load_state_dict(checkpoint["model"])
        result[name] = {
            "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
            **study.evaluate(model, "u3net", split, study.BATCH["u3net"]),
        }
        del model
        torch.cuda.empty_cache()
    result["dataset"] = "data/GFS128/test_clean.h5"
    result["samples"] = len(split[0])
    result["noise_condition"] = "clean (snr=+inf, model std=0)"
    output = RUN / "test_clean_evaluation.json"
    study.dump(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
