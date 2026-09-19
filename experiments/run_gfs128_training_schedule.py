"""Run the GFS128 official-source models in bounded concurrent lanes.

Only non-diffusion models are launched here.  HF diffusion is referenced as an
already completed, from-scratch run; no other diffusion implementation is allowed
by this schedule.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "results" / "gfs128_upstream"
TRAIN = ROOT / "experiments" / "train_gfs128_upstream.py"
SQD = ROOT / "experiments" / "train_gfs128_sqdlstm_torch.py"
STATUS = OUT / "training_schedule_status.json"
PLAN = OUT / "training_schedule.json"

# The largest observed simultaneous pair is PUNet (7.1 GiB) plus Restormer
# (11.6 GiB), with SQD-LSTM using another 0.46 GiB.  This remains below the
# 4090's 24 GiB while preserving every official batch size.
LANES = {
    "recurrent": ["sqd_lstm_torch"],
    "cnn": ["dlpu", "punet", "vurnet"],
    "transformer": ["uformer", "restormer"],
}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def completed(method: str) -> bool:
    return (OUT / "runs" / method / "complete.json").exists()


def command(method: str) -> list[str]:
    if method == "sqd_lstm_torch":
        return [sys.executable, "-B", "-u", str(SQD), "train"]
    return [sys.executable, "-B", "-u", str(TRAIN), "train", "--method", method]


def write_status(state: str, active: dict, finished: dict, started: float) -> None:
    dump(STATUS, {
        "state": state,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "active": {lane: {"method": x["method"], "pid": x["process"].pid} for lane, x in active.items()},
        "finished": finished,
        "lanes": LANES,
        "diffusion": {"planned": ["hf_diffusion"], "excluded": ["chen_full_hf", "fdu", "ddc", "wwfca"]},
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dump(PLAN, {
        "dataset": "GFS128",
        "gpu": "NVIDIA GeForce RTX 4090 24GB",
        "policy": "bounded three-lane concurrency; official hyperparameters unchanged",
        "lanes": LANES,
        "resource_profile": "resource_profile.json",
        "timing_policy": "Concurrent epoch times are operational only; report isolated fixed-batch benchmarks for method comparisons.",
        "diffusion_policy": "HF diffusion only",
        "hf_diffusion": {
            "action": "train from scratch in the diffusion schedule",
            "run": "../gfs_rme128/runs/GFS128/hf_base",
            "epochs": 300,
            "parameters": 1046017,
            "cross_attention": False,
        },
    })

    queues = {lane: list(methods) for lane, methods in LANES.items()}
    active: dict[str, dict] = {}
    finished: dict[str, dict] = {}
    started = time.time()

    while queues or active:
        for lane in list(queues):
            if lane in active:
                continue
            while queues[lane] and completed(queues[lane][0]):
                method = queues[lane].pop(0)
                finished[method] = {"state": "already_complete", "exit_code": 0}
            if not queues[lane]:
                del queues[lane]
                continue
            method = queues[lane].pop(0)
            log = (OUT / f"{method}.log").open("a", encoding="utf-8", buffering=1)
            err = (OUT / f"{method}.err.log").open("a", encoding="utf-8", buffering=1)
            log.write(f"\n=== scheduled {time.strftime('%Y-%m-%d %H:%M:%S')} lane={lane} ===\n")
            proc = subprocess.Popen(command(method), cwd=ROOT, stdout=log, stderr=err)
            active[lane] = {"method": method, "process": proc, "log": log, "err": err, "started": time.time()}

        write_status("training", active, finished, started)
        time.sleep(10)

        for lane, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close(); item["err"].close()
            method = item["method"]
            finished[method] = {
                "state": "complete" if code == 0 and completed(method) else "failed",
                "exit_code": code,
                "seconds": time.time() - item["started"],
            }
            del active[lane]
            if code != 0:
                # A failed lane does not launch its next model; independent lanes
                # continue so one repository cannot discard other completed work.
                queues.pop(lane, None)

    failed = [m for m, x in finished.items() if x["state"] == "failed"]
    write_status("failed" if failed else "complete", active, finished, started)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
