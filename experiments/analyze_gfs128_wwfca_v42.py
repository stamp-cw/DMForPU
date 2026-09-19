"""Compare constant, timestep-only, and timestep+noise WWFCA residual gates."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/"experiments/results/gfs_rme128/runs/GFS128";OUT=ROOT/"experiments/results/gfs128_wwfca_v42"
METHODS=("wwfca_v41","wwfca_v42_time","wwfca_v42")
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);histories={m:json.loads((RUNS/m/"history.json").read_text()) for m in METHODS};rows=[];tests=[]
    for m,h in histories.items():
        best=min(h,key=lambda r:r["val_u3_aligned_nrmse"]);rows.append({"method":m,"selected_epoch":best["epoch"],**{k:v for k,v in best.items() if k.startswith("val_")}})
        complete=json.loads((RUNS/m/"complete.json").read_text())
        for condition,values in complete["tests"].items():tests.append({"method":m,"condition":condition,"selected_epoch":complete["selected_epoch"],**values})
    write_csv(OUT/"primary_validation_metrics.csv",rows);write_csv(OUT/"test_metrics.csv",tests)
    panels=(("val_u3_aligned_nrmse","U3-aligned NRMSE"),("gamma","Mean gamma"),("gamma_min","Min gamma"),("gamma_max","Max gamma"),("time_strength","Time strength"),("noise_strength","Noise strength"))
    fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True);colors={METHODS[0]:"#377eb8",METHODS[1]:"#4daf4a",METHODS[2]:"#e41a1c"}
    for ax,(key,title) in zip(axes.flat,panels):
        for m,h in histories.items():ax.plot([r["epoch"] for r in h],[r.get(key,r.get("gamma",0) if key in ("gamma_min","gamma_max") else 0) for r in h],label=m,color=colors[m])
        ax.set_title(title);ax.set_xlabel("Epoch");ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle("GFS128 WWFCA timestep/noise gate ablation");fig.savefig(OUT/"training_diagnostics.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
