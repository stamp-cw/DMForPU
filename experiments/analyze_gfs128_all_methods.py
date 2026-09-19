"""Evaluate validation-best checkpoints and visualize every GFS128 method."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
BASE=ROOT/"experiments/results/gfs128_all_methods";BASE.mkdir(parents=True,exist_ok=True)
UP=ROOT/"experiments/results/gfs128_upstream/runs"
DIFF=ROOT/"experiments/results/gfs_rme128/runs/GFS128"
SNRS=(0,5,10,20,30)
UPSTREAM=("u3net","dlpu","punet","uformer","restormer","vurnet")
DIFFUSION=("hf_base","chen_physics","chen_sr","chen_physics_sr","chen_full","chen_no_sparse","chen_no_cam",
           "hf_matched","dcc_only","wwfca_only","fdu")
LABEL={"u3net":"U3Net","dlpu":"DLPU","punet":"PUNet","uformer":"Uformer","restormer":"Restormer",
       "vurnet":"VUR-Net",
       "sqd_lstm_torch":"SQD-LSTM","hf_base":"HF diffusion","chen_physics":"Chen physics","chen_sr":"Chen SR",
       "chen_physics_sr":"Chen physics+SR","chen_full":"Chen full","chen_no_sparse":"Chen no-sparse",
       "chen_no_cam":"Chen no-CAM","hf_matched":"HF matched","dcc_only":"DCC only",
       "wwfca_only":"WWFCA only","fdu":"DCC+WWFCA"}


def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def evaluate_upstream():
    import experiments.train_gfs128_upstream as study
    _,_,tests=study.load_data();results={}
    with h5py.File(ROOT/"data/GFS128/test_clean.h5","r") as f:
        clean=(torch.from_numpy(f["psi"][:])[:,None],torch.from_numpy(f["phi"][:])[:,None],torch.from_numpy(f["snr"][:]))
    for method in UPSTREAM:
        output=UP/method/"best_evaluation.json"
        if output.exists():results[method]=json.loads(output.read_text(encoding="utf-8"));continue
        state=torch.load(UP/method/"best.pth",map_location="cpu",weights_only=False);model=study.build(method);model.load_state_dict(state["model"])
        item={"method":method,"selection":"validation-best","selected_epoch":state["epoch"]+1,
              "parameters":sum(p.numel() for p in model.parameters()),"tests":{}}
        for snr in SNRS:item["tests"][str(snr)]=study.evaluate(model,method,tests[snr],study.BATCH[method])
        item["tests"]["clean"]=study.evaluate(model,method,clean,study.BATCH[method]);dump(output,item);results[method]=item
        del model;torch.cuda.empty_cache()
    return results


def evaluate_sqd():
    from model.sqdlstm_torch import JointConvSQDLSTMNetTorch
    from experiments.train_gfs128_sqdlstm_torch import evaluate_file
    output=UP/"sqd_lstm_torch/best_evaluation.json"
    if output.exists():return json.loads(output.read_text(encoding="utf-8"))
    state=torch.load(UP/"sqd_lstm_torch/best.pth",map_location="cpu",weights_only=False);model=JointConvSQDLSTMNetTorch().cuda();model.load_state_dict(state["model"])
    item={"method":"sqd_lstm_torch","selection":"official train-loss best","selected_epoch":state["epoch"],
          "parameters":model.trainable_parameters,"tests":{}}
    for snr in ("clean",*SNRS):
        name="test_clean.h5" if snr=="clean" else f"test_{snr}dB.h5"
        item["tests"][str(snr)]=evaluate_file(model,ROOT/"data/GFS128"/name)
    dump(output,item);return item


def load_results():
    results=evaluate_upstream();results["sqd_lstm_torch"]=evaluate_sqd()
    for method in DIFFUSION:
        path=DIFF/method/"complete.json"
        if not path.exists():raise FileNotFoundError(path)
        results[method]=json.loads(path.read_text(encoding="utf-8"))
    return results


def rows_from(results):
    rows=[]
    for method,item in results.items():
        for snr in SNRS:
            metrics=item["tests"][str(snr)]
            rows.append({"method":LABEL[method],"family":"diffusion" if method in DIFFUSION else "deep",
                         "snr_db":snr,"mean_aligned_mae":metrics["mean_aligned_mae"],
                         "mean_aligned_nrmse":metrics["mean_aligned_nrmse"],"pge":metrics.get("pge"),
                         "u3_aligned_mae":metrics["u3_aligned_mae"],
                         "u3_aligned_rmse":metrics["u3_aligned_rmse"],
                         "u3_aligned_nrmse":metrics["u3_aligned_nrmse"],
                         "u3_aligned_ssim":metrics["u3_aligned_ssim"],
                         "raw_au":metrics["raw_au"],
                         "integer_aligned_au":metrics["integer_aligned_au"],
                         "range_aligned_au":metrics["range_aligned_au"],
                         "parameters":item.get("parameters",0),"selected_epoch":item.get("selected_epoch")})
    traditional=list(csv.DictReader((ROOT/"experiments/results/gfs_rme128/traditional_lsqgdct/summary.csv").open(encoding="utf-8-sig")))
    for source in traditional:
        if source["dataset"]=="GFS128" and int(source["snr"]) in SNRS:
            rows.append({"method":source["method"],"family":"traditional","snr_db":int(source["snr"]),
                         "mean_aligned_mae":float(source["mean_aligned_mae"]),"mean_aligned_nrmse":float(source["mean_aligned_nrmse"]),
                         "u3_aligned_mae":float(source["u3_aligned_mae"]),
                         "u3_aligned_rmse":float(source["u3_aligned_rmse"]),
                         "u3_aligned_nrmse":float(source["u3_aligned_nrmse"]),
                         "u3_aligned_ssim":float(source["u3_aligned_ssim"]),
                         "raw_au":float(source["raw_au"]),
                         "integer_aligned_au":float(source["integer_aligned_au"]),
                         "range_aligned_au":float(source["range_aligned_au"]),
                         "pge":float(source["pge"]),"parameters":0,"selected_epoch":None})
    return rows


def plot_group(rows,names,title,path):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for name in names:
        selected=sorted((r for r in rows if r["method"]==name),key=lambda x:x["snr_db"])
        if not selected:continue
        x=[r["snr_db"] for r in selected]
        axes[0,0].plot(x,[r["mean_aligned_mae"] for r in selected],marker="o",label=name)
        axes[0,1].plot(x,[r["u3_aligned_mae"] for r in selected],marker="o",label=name)
        axes[1,0].plot(x,[r["u3_aligned_nrmse"] for r in selected],marker="o",label=name)
        axes[1,1].plot(x,[r["u3_aligned_ssim"] for r in selected],marker="o",label=name)
    axes[0,0].set_ylabel("Mean-aligned MAE (rad)");axes[0,1].set_ylabel("U3-aligned MAE (rad)")
    axes[1,0].set_ylabel("U3-aligned NRMSE");axes[1,1].set_ylabel("U3-aligned SSIM")
    for axis in axes.flat:axis.set_xlabel("SNR (dB)");axis.grid(alpha=.25);axis.legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)


def plot_au(rows,names,path):
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    fields=(("raw_au","Raw AU (%)"),("integer_aligned_au","I-AU (%)"),
            ("range_aligned_au","RA-AU (%)"))
    for name in names:
        selected=sorted((r for r in rows if r["method"]==name),key=lambda x:x["snr_db"])
        if not selected:continue
        for axis,(field,label) in zip(axes,fields):
            axis.plot([r["snr_db"] for r in selected],[r[field] for r in selected],marker="o",label=name)
            axis.set_ylabel(label)
    for axis in axes:
        axis.set_xlabel("SNR (dB)");axis.set_ylim(0,100);axis.grid(alpha=.25)
        axis.legend(frameon=False,fontsize=7,ncol=2)
    fig.suptitle("GFS128: Accuracy of Unwrapping")
    fig.tight_layout();fig.savefig(path,dpi=200);plt.close(fig)


def main():
    results=load_results();rows=rows_from(results)
    with (BASE/"all_methods.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(BASE/"all_methods.json",rows)
    plot_group(rows,["LS","QG","DCT","DLPU","PUNet","VUR-Net","SQD-LSTM","U3Net","Uformer","Restormer","HF diffusion","Chen full","DCC+WWFCA"],
               "GFS128: all principal methods",BASE/"all_principal_methods.png")
    plot_group(rows,["HF matched","DCC only","WWFCA only","DCC+WWFCA"],"GFS128: DCC × WWFCA ablation",BASE/"dcc_wwfca_ablation.png")
    plot_group(rows,["HF diffusion","Chen physics","Chen SR","Chen physics+SR","Chen full","Chen no-sparse","Chen no-CAM"],
               "GFS128: Chen component ablation",BASE/"chen_ablation.png")
    plot_au(rows,["LS","QG","DCT","DLPU","PUNet","VUR-Net","SQD-LSTM","U3Net","Uformer","Restormer","HF diffusion","Chen full","DCC+WWFCA"],
            BASE/"au_comparison.png")
    averages=[]
    for name in sorted(set(r["method"] for r in rows)):
        chosen=[r["mean_aligned_mae"] for r in rows if r["method"]==name];averages.append((name,float(np.mean(chosen))))
    averages.sort(key=lambda x:x[1]);fig,ax=plt.subplots(figsize=(11,7));ax.barh([x[0] for x in averages],[x[1] for x in averages]);ax.invert_yaxis()
    ax.set_xlabel("Mean-aligned MAE averaged over 0/5/10/20/30 dB (rad)");ax.grid(axis="x",alpha=.25);fig.tight_layout();fig.savefig(BASE/"average_mae_ranking.png",dpi=200);plt.close(fig)
    print(BASE)


if __name__=="__main__":main()
