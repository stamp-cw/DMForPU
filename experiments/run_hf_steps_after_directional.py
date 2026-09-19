"""Wait for the active directional ablation, then run the HF step benchmark."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIT_STATUS = ROOT / "experiments" / "results" / "gfs128_directional_hf_ablation" / "schedule_status.json"
OUT = ROOT / "experiments" / "results" / "hf_inference_steps_gfs128"


def status(state, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "queue_status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"state": state, "updated_at": time.strftime("%F %T"), **extra}, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    status("waiting_for_directional_ablation")
    while True:
        try:
            current = json.loads(WAIT_STATUS.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        if current.get("state") != "training":
            break
        time.sleep(20)
    status("running", prerequisite_state=current.get("state", "unknown"))
    log_path = OUT / "benchmark.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        result = subprocess.run([sys.executable, "-B", "-u", str(ROOT / "experiments" / "evaluate_hf_inference_steps_gfs128.py")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status("complete" if result.returncode == 0 else "failed", returncode=result.returncode)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
