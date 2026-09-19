"""Compare off, directional-only, and directional+attention v4.1 variants."""
from __future__ import annotations
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/"experiments/results/gfs_rme128/runs/GFS128";OUT=ROOT/"experiments/results/gfs128_directional_hf_ablation"
METHODS=("wwfca_v41_off","wwfca_v41_noattn","wwfca_v41")
def csv_write(path,rows):
    with path.open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);histories={m:json.loads((RUNS/m/"history.json").read_text()) for m in METHODS};summary=[];tests=[]
    for m,h in histories.items():
        b=min(h,key=lambda r:r["val_u3_aligned_nrmse"]);summary.append({"method":m,"selected_epoch":b["epoch"],**{k:v for k,v in b.items() if k.startswith("val_")}})
        complete=json.loads((RUNS/m/"complete.json").read_text())
        for c,v in complete["tests"].items():tests.append({"method":m,"condition":c,"selected_epoch":complete["selected_epoch"],**v})
    csv_write(OUT/"validation_metrics.csv",summary);csv_write(OUT/"test_metrics.csv",tests)
    panels=(("val_u3_aligned_nrmse","U3-aligned NRMSE"),("val_mean_aligned_nrmse","Mean-aligned NRMSE"),("val_u3_aligned_ssim","U3-aligned SSIM"),("gamma","Gamma"),("residual_rms_ratio","Injected/base RMS"),("seconds","Epoch seconds"))
    colors={METHODS[0]:"#377eb8",METHODS[1]:"#4daf4a",METHODS[2]:"#e41a1c"};fig,axes=plt.subplots(2,3,figsize=(18,10),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,panels):
        for m,h in histories.items():ax.plot([r["epoch"] for r in h],[r.get(key,0) for r in h],label=m,color=colors[m])
        ax.set_title(title);ax.set_xlabel("Epoch");ax.grid(alpha=.25)
    axes[0,0].legend();fig.suptitle("GFS128 directional high-frequency residual ablation");fig.savefig(OUT/"training_diagnostics.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
