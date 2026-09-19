"""Aggregate GFS128/RME128 results, classical baselines, and figures."""
from __future__ import annotations
import csv,json,sys,time
from pathlib import Path
import h5py,numpy as np,torch,matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiments.train_gfs_rme128 as study
from traditional.phase_unwrapping import METHODS as CLASSICAL

OUT=study.OUT;FIG=OUT/"figures";FIG.mkdir(parents=True,exist_ok=True)
LABEL={"dlpu":"DLPU","punet":"PUNet","u3net":"U3Net","hf_base":"HF diffusion",
       "chen_full":"Chen-full HF","sqd_lstm":"SQD-LSTM","restormer":"Restormer","uformer":"Uformer"}


def load_best(dataset,method):
    folder=OUT/"runs"/dataset/method;st=torch.load(folder/"best.pth",map_location="cpu",weights_only=False)
    if method in ("hf_base","chen_full"):model,_=study.diffusion_model(method)
    else:model=study.non_diff_model(method)
    model.load_state_dict(st["model"]);return model.eval(),st["epoch"]+1


def classical():
    output=OUT/"traditional.json"
    if output.exists():return json.loads(output.read_text(encoding="utf-8"))
    summaries={};detail=[]
    for dataset in study.DATASETS:
        summaries[dataset]={}
        for snr in study.TEST_SNRS:
            with h5py.File(ROOT/"data"/dataset/f"test_{snr}dB.h5","r") as f:w=f["psi"][:];t=f["phi"][:]
            for name,fn in CLASSICAL.items():
                sums={};begin=time.perf_counter()
                for i,(wi,ti) in enumerate(zip(w,t)):
                    p=fn(wi);m=study.metric_sums(torch.from_numpy(p)[None,None],torch.from_numpy(ti)[None,None],torch.from_numpy(wi)[None,None])
                    detail.append({"dataset":dataset,"snr":snr,"sample":i,"method":name,**m})
                    for k,v in m.items():sums[k]=sums.get(k,0.)+v
                summaries[dataset].setdefault(name,{})[str(snr)]={k:v/len(w) for k,v in sums.items()}
                summaries[dataset][name][str(snr)]["latency_ms"]=(time.perf_counter()-begin)*1000/len(w)
                print(f"traditional {dataset} {snr} {name}",flush=True)
    with (OUT/"traditional_per_image.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=list(detail[0]));wr.writeheader();wr.writerows(detail)
    study.dump(output,summaries);return summaries


def qualitative(dataset,snr,index=0):
    with h5py.File(ROOT/"data"/dataset/f"test_{snr}dB.h5","r") as f:
        w=torch.from_numpy(f["psi"][index:index+1,None]).cuda();t=f["phi"][index];s=torch.from_numpy(f["snr"][index:index+1]).cuda()
    std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10));preds=[]
    for method in study.METHODS:
        model,_=load_best(dataset,method);gen=torch.Generator(device="cuda").manual_seed(30000+snr)
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16 if method in ("hf_base","chen_full") else torch.bfloat16,enabled=method!="u3net"):
            p=study.infer(model,method,w,std,gen)[0,0].float().cpu().numpy()
        p=p-(p-t).mean();preds.append((LABEL[method],p));del model;torch.cuda.empty_cache()
    wn=w[0,0].cpu().numpy()
    for name,fn in CLASSICAL.items():
        p=fn(wn);p=p-(p-t).mean();preds.append((name,p))
    fig,axes=plt.subplots(3,4,figsize=(13,10),constrained_layout=True);items=[("Ground truth",t)]+preds
    limit=max(np.quantile(np.abs(p-t),.98) for _,p in preds)
    for ax,(name,p) in zip(axes.flat,items):
        im=ax.imshow(np.abs(p-t) if name!="Ground truth" else p,cmap="magma" if name!="Ground truth" else "viridis",
                     vmin=0 if name!="Ground truth" else None,vmax=limit if name!="Ground truth" else None)
        ax.set_title(name);ax.axis("off")
    for ax in axes.flat[len(items):]:ax.axis("off")
    fig.suptitle(f"{dataset}, {snr} dB: mean-aligned absolute errors");fig.savefig(FIG/f"qualitative_{dataset}_{snr}dB.png",dpi=180);plt.close(fig)


def main():
    missing=[f"{d}/{m}" for d in study.DATASETS for m in study.METHODS if not (OUT/"runs"/d/m/"complete.json").exists()]
    if missing:raise RuntimeError(f"unfinished: {missing}")
    trad=classical();rows=[]
    for dataset in study.DATASETS:
        for method in study.METHODS:
            r=json.loads((OUT/"runs"/dataset/method/"complete.json").read_text(encoding="utf-8"))
            for snr,v in r["tests"].items():rows.append({"dataset":dataset,"method":LABEL[method],"snr":int(snr),"selected_epoch":r["selected_epoch"],"parameters":r["parameters"],"training_hours":r["total_seconds"]/3600,**v})
        for method,levels in trad[dataset].items():
            for snr,v in levels.items():rows.append({"dataset":dataset,"method":method,"snr":int(snr),"selected_epoch":0,"parameters":0,"training_hours":0.,**v})
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    study.dump(OUT/"comparison.json",rows)
    for dataset in study.DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        methods=list(LABEL.values())+list(CLASSICAL)
        for name in methods:
            rr=sorted([x for x in rows if x["dataset"]==dataset and x["method"]==name],key=lambda x:x["snr"])
            axes[0].plot([x["snr"] for x in rr],[x["mean_aligned_mae"] for x in rr],"o-",label=name)
            axes[1].plot([x["snr"] for x in rr],[100*x["mean_aligned_nrmse"] for x in rr],"o-",label=name)
        axes[0].set(xlabel="SNR (dB)",ylabel="Mean-aligned MAE (rad)");axes[1].set(xlabel="SNR (dB)",ylabel="NRMSE (%)")
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
        fig.savefig(FIG/f"noise_curves_{dataset}.png",dpi=190);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
        for method in study.METHODS:
            h=json.loads((OUT/"runs"/dataset/method/"history.json").read_text(encoding="utf-8"));valid=[x for x in h if x["val_mean_aligned_mae"] is not None]
            axes[0].plot([x["epoch"] for x in h],[x.get("train_loss",x.get("train_total")) for x in h],label=LABEL[method])
            axes[1].plot([x["epoch"] for x in valid],[x["val_mean_aligned_mae"] for x in valid],label=LABEL[method])
        axes[0].set(yscale="log",xlabel="Epoch",ylabel="Training loss");axes[1].set(yscale="log",xlabel="Epoch",ylabel="Validation MAE")
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7,ncol=2)
        fig.savefig(FIG/f"training_curves_{dataset}.png",dpi=190);plt.close(fig)
        for snr in (0,10,30):qualitative(dataset,snr)
    lines=["# GFS128 与 RME128 从头训练对比","","所有深度模型分别在每个数据族从头训练；4,500 张训练、500 张验证，测试集为每个 SNR 独立的 1,000 张。下表为 mean-aligned MAE。","",
           "| 数据 | 方法 | 0 dB | 5 dB | 10 dB | 20 dB | 30 dB |","|---|---|---:|---:|---:|---:|---:|"]
    for dataset in study.DATASETS:
        for name in list(LABEL.values())+list(CLASSICAL):
            rr={x["snr"]:x for x in rows if x["dataset"]==dataset and x["method"]==name};lines.append(f"| {dataset} | {name} | "+" | ".join(f"{rr[s]['mean_aligned_mae']:.4f}" for s in study.TEST_SNRS)+" |")
    lines += ["","完整指标见 `comparison.csv`；逐轮曲线与定性误差图见 `figures/`。HF diffusion 与 Chen-full 都采用无交叉注意力的 Hugging Face UNet2DModel，Chen-full 额外包含物理迭代、CAM、稀疏误差、配对再扰动和自蒸馏；WWFCA 变体才使用交叉注意力。"]
    (OUT/"REPORT.zh-CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":main()
