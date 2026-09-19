"""Create v4.1/off paired tables, figures, and comparisons with v4."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/"experiments/results/gfs_rme128/runs/GFS128"
OUT=ROOT/"experiments/results/gfs128_wwfca_v41";METHODS=("wwfca_v41_off","wwfca_v41")
HIGHER={"u3_aligned_ssim","raw_au","integer_aligned_au","range_aligned_au"}

def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    histories={m:json.loads((RUNS/m/"history.json").read_text()) for m in METHODS};common=min(map(len,histories.values()))
    metrics=[k[4:] for k in histories[METHODS[0]][0] if k.startswith("val_")];selected=[];extrema=[]
    for method,history in histories.items():
        rows=history[:common];best=min(rows,key=lambda r:r["val_u3_aligned_nrmse"])
        selected.append({"method":method,"common_epochs":common,"selected_epoch":best["epoch"],
                         **{m:best["val_"+m] for m in metrics}})
        for metric in metrics:
            row=(max(rows,key=lambda r:r["val_"+metric]) if metric in HIGHER else min(rows,key=lambda r:r["val_"+metric]))
            extrema.append({"method":method,"metric":metric,"direction":"max" if metric in HIGHER else "min",
                            "epoch":row["epoch"],"value":row["val_"+metric]})
    write_csv(OUT/"paired_primary_metrics.csv",selected);write_csv(OUT/"paired_per_metric_best.csv",extrema)
    (OUT/"paired_summary.json").write_text(json.dumps({"common_epochs":common,"selected":selected,
        "per_metric_best":extrema},indent=2),encoding="utf-8")
    panels=(("val_u3_aligned_nrmse","U3-aligned NRMSE"),("val_u3_aligned_ssim","U3-aligned SSIM"),
            ("residual_rms_ratio","Injected/base RMS"),("attention_entropy","Attention entropy"),
            ("gate_spatial_std","Gate spatial std"),("temperature","Attention temperature"))
    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True);colors={METHODS[0]:"#377eb8",METHODS[1]:"#d95f02"}
    for ax,(key,title) in zip(axes.flat,panels):
        for method,history in histories.items():ax.plot([r["epoch"] for r in history[:common]],[r.get(key,0) for r in history[:common]],label=method,color=colors[method])
        ax.set_title(title);ax.set_xlabel("Epoch");ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle("GFS128 WWFCA-v4.1 controlled ablation");fig.savefig(OUT/"paired_training_diagnostics.png",dpi=180);plt.close(fig)
    tests=[]
    for method in METHODS:
        complete=json.loads((RUNS/method/"complete.json").read_text())
        for condition,values in complete["tests"].items():tests.append({"method":method,"condition":condition,"selected_epoch":complete["selected_epoch"],**values})
    write_csv(OUT/"paired_test_metrics.csv",tests)

if __name__=="__main__":main()
