"""Evaluate LS, QG and Schofield-DCT on all current GFS128 test conditions."""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate_gfs_rme_traditional import metrics
from traditional.phase_unwrapping import unwrap_dct_schofield, unwrap_least_squares, unwrap_quality_guided

OUT = ROOT / "experiments/results/gfs128_traditional"
SHARDS = OUT / "shards"
FIGURES = OUT / "figures"
CONDITIONS = ("clean", "0", "5", "10", "20", "30")
METHODS = {"LS": unwrap_least_squares, "QG": unwrap_quality_guided, "DCT": unwrap_dct_schofield}
METRICS = ("raw_mae", "raw_rmse", "integer_aligned_mae", "integer_aligned_rmse",
           "mean_aligned_mae", "mean_aligned_rmse", "mean_aligned_nrmse", "pge",
           "rewrap_circular_mae", "u3_aligned_mae", "u3_aligned_rmse",
           "u3_aligned_nrmse", "u3_aligned_ssim", "raw_au", "integer_aligned_au", "range_aligned_au")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"); tmp.replace(path)


def data_path(condition):
    return ROOT / "data/GFS128" / ("test_clean.h5" if condition == "clean" else f"test_{condition}dB.h5")


def chunk(task):
    condition, method, begin, end = task; fn = METHODS[method]
    with h5py.File(data_path(condition), "r") as f:
        wrapped, target, scene_ids = f["psi"][begin:end], f["phi"][begin:end], f["scene_id"][begin:end]
    rows = []
    for offset, (w, t, scene_id) in enumerate(zip(wrapped, target, scene_ids)):
        wall = time.perf_counter(); cpu = time.process_time(); prediction = fn(w)
        cpu_ms = (time.process_time()-cpu)*1000; wall_ms = (time.perf_counter()-wall)*1000
        if prediction.shape != t.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"invalid {condition}/{method}/{begin+offset}")
        rows.append({"condition": condition, "method": method, "sample": begin+offset,
                     "scene_id": int(scene_id), "algorithm_wall_ms": wall_ms,
                     "algorithm_cpu_ms": cpu_ms, **metrics(prediction, t, w)})
    return rows


def evaluate(pool, condition, method, workers):
    shard = SHARDS / f"{condition}_{method}.json"
    if shard.exists(): return json.loads(shard.read_text(encoding="utf-8"))
    bounds = np.linspace(0, 1000, workers+1, dtype=int)
    tasks = [(condition, method, int(bounds[i]), int(bounds[i+1])) for i in range(workers)]
    start = time.perf_counter(); rows = [row for part in pool.map(chunk, tasks) for row in part]
    rows.sort(key=lambda row: row["sample"]); elapsed = time.perf_counter()-start
    payload = {"condition": condition, "method": method, "samples": len(rows),
               "evaluation_wall_seconds": elapsed, "workers": workers, "rows": rows}
    dump(shard, payload)
    print(f"{condition:>5} {method}: {elapsed:.2f}s | U3-NRMSE={np.mean([r['u3_aligned_nrmse'] for r in rows]):.5f}", flush=True)
    return payload


def aggregate(payloads):
    summary=[]
    for payload in payloads:
        rows=payload["rows"]
        item={"dataset":"GFS128","condition":payload["condition"],"method":payload["method"],
              "samples":len(rows),"parameters":0,"training_required":False,
              "evaluation_wall_seconds":payload["evaluation_wall_seconds"],
              "evaluation_throughput_images_s":len(rows)/payload["evaluation_wall_seconds"]}
        for key in (*METRICS,"algorithm_wall_ms","algorithm_cpu_ms"):
            values=np.asarray([row[key] for row in rows],np.float64)
            item[key]=float(values.mean());item[key+"_std"]=float(values.std(ddof=1))
        summary.append(item)
    return summary


def write_csv(path, rows):
    with path.with_suffix(path.suffix+".tmp").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.with_suffix(path.suffix+".tmp").replace(path)


def plots(summary):
    FIGURES.mkdir(parents=True,exist_ok=True);colors={"LS":"#3178b5","QG":"#d05f2d","DCT":"#288a68"};x=np.arange(len(CONDITIONS))
    fields=(("u3_aligned_nrmse","U3-aligned NRMSE"),("u3_aligned_ssim","U3-aligned SSIM"),
            ("mean_aligned_mae","Mean-aligned MAE (rad)"),("range_aligned_au","RA-AU (%)"))
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for method in METHODS:
        rows={r["condition"]:r for r in summary if r["method"]==method}
        for ax,(field,label) in zip(axes.flat,fields):
            ax.plot(x,[rows[c][field] for c in CONDITIONS],"o-",label=method,color=colors[method]);ax.set_ylabel(label)
    for ax in axes.flat:ax.set_xticks(x,CONDITIONS);ax.set_xlabel("condition / dB");ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.suptitle("GFS128 traditional phase-unwrapping methods");fig.tight_layout();fig.savefig(FIGURES/"metric_comparison.png",dpi=220);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,5));width=.24
    for i,method in enumerate(METHODS):
        rows={r["condition"]:r for r in summary if r["method"]==method}
        ax.bar(x+(i-1)*width,[rows[c]["algorithm_wall_ms"] for c in CONDITIONS],width,label=method,color=colors[method])
    ax.set_xticks(x,CONDITIONS);ax.set_ylabel("CPU wall time (ms/image)");ax.set_xlabel("condition / dB");ax.grid(axis="y",alpha=.25);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(FIGURES/"runtime_comparison.png",dpi=220);plt.close(fig)

    fig,axes=plt.subplots(3,5,figsize=(14,8.5))
    for row,condition in enumerate(("clean","0","10")):
        with h5py.File(data_path(condition),"r") as f:w,t=f["psi"][0],f["phi"][0]
        preds={name:fn(w) for name,fn in METHODS.items()};axes[row,0].imshow(w,cmap="twilight_shifted",vmin=-math.pi,vmax=math.pi);axes[row,1].imshow(t,cmap="turbo")
        axes[row,0].set_ylabel(condition,rotation=0,labelpad=20)
        for col,(name,pred) in enumerate(preds.items(),2):
            aligned=pred-(pred-t).mean();axes[row,col].imshow(np.abs(aligned-t),cmap="magma",vmin=0,vmax=np.quantile(np.abs(aligned-t),.99))
            if row==0:axes[row,col].set_title(f"{name} abs. error")
        if row==0:axes[row,0].set_title("Wrapped");axes[row,1].set_title("Ground truth")
    for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle("GFS128 qualitative comparison, sample 0");fig.tight_layout();fig.savefig(FIGURES/"qualitative_comparison.png",dpi=220);plt.close(fig)


def report(summary):
    rows=[]
    for method in METHODS:
        selected={r["condition"]:r for r in summary if r["method"]==method}
        rows.append({"method":method,**{c:selected[c]["u3_aligned_nrmse"] for c in CONDITIONS},
                     "average":float(np.mean([selected[c]["u3_aligned_nrmse"] for c in CONDITIONS])),
                     "cpu_ms":float(np.mean([selected[c]["algorithm_wall_ms"] for c in CONDITIONS]))})
    lines=["# GFS128 traditional phase-unwrapping results","",
           "The primary table reports U3Net range-aligned NRMSE over all 1,000 test images per condition.","",
           "| Method | clean | 0 dB | 5 dB | 10 dB | 20 dB | 30 dB | Six-condition mean | CPU ms/image |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['method']} | "+" | ".join(f"{r[c]:.5f}" for c in CONDITIONS)+f" | {r['average']:.5f} | {r['cpu_ms']:.2f} |")
    lines += ["","- LS: unweighted gradient-domain least squares with a Neumann Poisson solver.",
              "- QG: second-difference reliability-guided maximum spanning tree.",
              "- DCT: Schofield sin/cos Laplacian with a DCT Neumann solver and integer-cycle projection.",
              "- All methods have zero trainable parameters and require no training.",
              "- Per-image metrics, all alignment variants, SSIM, AU, and timing are retained in CSV and JSON files."]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    workers=max(1,min(4,(os.cpu_count() or 4)//2));OUT.mkdir(parents=True,exist_ok=True);SHARDS.mkdir(exist_ok=True)
    dump(OUT/"protocol.json",{"dataset":"GFS128","conditions":list(CONDITIONS),"methods":list(METHODS),
         "samples_per_condition":1000,"workers":workers,"platform":platform.platform(),
         "metrics":"same unified definitions as neural methods","parameters":0,"training_required":False})
    payloads=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for condition in CONDITIONS:
            for method in METHODS:
                payloads.append(evaluate(pool,condition,method,workers))
                dump(OUT/"status.json",{"state":"running","completed_splits":len(payloads),"total_splits":18,
                                        "condition":condition,"method":method})
    summary=aggregate(payloads);all_rows=[row for payload in payloads for row in payload["rows"]]
    write_csv(OUT/"summary.csv",summary);write_csv(OUT/"per_image.csv",all_rows);dump(OUT/"summary.json",summary)
    plots(summary);report(summary);dump(OUT/"status.json",{"state":"complete","evaluations":len(all_rows),"summary_rows":len(summary)})
    print(OUT)


if __name__=="__main__":main()
