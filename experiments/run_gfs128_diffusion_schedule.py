"""Deferred, resumable GFS128 diffusion suite followed by unified analysis."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/results/gfs128_diffusion_suite"
TRAIN=ROOT/"experiments/train_gfs_rme128.py"
MAIN=ROOT/"experiments/results/gfs128_upstream/runs"
METHODS=("hf_base","chen_full","chen_physics","chen_sr","chen_physics_sr","chen_no_sparse","chen_no_cam",
         "hf_matched","dcc_only","wwfca_only","fdu")
MATCHED={"hf_matched","dcc_only","wwfca_only","fdu"}


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8");tmp.replace(path)


def status(state,**extra):
    dump(OUT/"schedule_status.json",{"state":state,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "methods":METHODS,"hf_base":"train from scratch without cross-attention",**extra})


def prerequisites_ready():
    # Wait for the two short lanes to release GPU compute. The transformer lane
    # may continue in parallel; only one diffusion model is ever active.
    return all((MAIN/m/"complete.json").exists() for m in ("punet","sqd_lstm_torch"))


def run_method(method):
    run=ROOT/"experiments/results/gfs_rme128/runs/GFS128"/method
    if (run/"complete.json").exists():return 0,"already_complete"
    log=(OUT/f"{method}.log").open("a",encoding="utf-8",buffering=1)
    err=(OUT/f"{method}.err.log").open("a",encoding="utf-8",buffering=1)
    log.write(f"\n=== scheduled {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    process=subprocess.Popen([sys.executable,"-B","-u",str(TRAIN),"train","--dataset","GFS128","--method",method],
                             cwd=ROOT,stdout=log,stderr=err)
    status("training",active=method,pid=process.pid)
    code=process.wait();log.close();err.close()
    return code,"complete" if code==0 and (run/"complete.json").exists() else "failed"


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/"plan.json",{
        "dataset":"GFS128","seed":42,"epochs":300,"batch":8,"selection":"best validation mean_aligned_mae",
        "queue":METHODS,"reuse":[],
        "matched_ablation":{"hf_matched":[0,0],"dcc_only":[1,0],"wwfca_only":[0,1],"fdu":[1,1]},
        "attention_ablation":{"hf_base":"HF UNet2DModel, no cross-attention","hf_matched":"matched HF UNet2DModel, no cross-attention","wwfca_only":"FDUNet with WWFCA cross-attention"},
        "chen_variants":["hf_base","chen_physics","chen_sr","chen_physics_sr","chen_full","chen_no_sparse","chen_no_cam"],
        "concurrency":"After PUNet/SQD-LSTM, run one small Chen model alongside the transformer lane; matched DCC/WWFCA models wait for Restormer.",
    })
    while not prerequisites_ready():
        status("waiting_for_punet_and_sqdlstm");time.sleep(30)
    finished={}
    for method in METHODS:
        if method in MATCHED:
            while not (MAIN/"restormer"/"complete.json").exists():
                status("waiting_for_restormer_before_large_diffusion",next_method=method,finished=finished);time.sleep(60)
        code,state=run_method(method);finished[method]={"state":state,"exit_code":code}
        status("training" if state!="failed" else "failed",finished=finished)
        if code:return code
    # Unified analysis waits for Restormer/Uformer as well.
    while not all((MAIN/m/"complete.json").exists() for m in ("u3net","dlpu","punet","uformer","restormer","vurnet","sqd_lstm_torch")):
        status("waiting_for_non_diffusion_completion",finished=finished);time.sleep(60)
    analyzer=ROOT/"experiments/analyze_gfs128_all_methods.py"
    code=subprocess.call([sys.executable,"-B",str(analyzer)],cwd=ROOT) if analyzer.exists() else 0
    status("complete" if code==0 else "analysis_failed",finished=finished,analysis_exit_code=code)
    return code


if __name__=="__main__":raise SystemExit(main())
