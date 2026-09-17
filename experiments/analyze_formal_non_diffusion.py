"""Aggregate metrics and generate figures after the formal non-diffusion queue."""
from __future__ import annotations

import csv, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import experiments.formal_non_diffusion_study as study
from traditional.phase_unwrapping import METHODS as TRADITIONAL

OUT=study.OUT; RUNS=OUT/"runs"; FIG=OUT/"figures"; FIG.mkdir(parents=True,exist_ok=True)
LABEL={"dlpu":"DLPU","sqd_lstm":"SQD-LSTM","punet":"PUNet","uformer":"Uformer-S","restormer":"Restormer","u3net":"U3Net"}


def np_metrics(pred,target,wrapped):
    k=np.round(np.median((target-pred).reshape(-1))/(2*np.pi)); err=pred+k*2*np.pi-target
    pge=np.mean(np.abs(np.concatenate((np.diff(err,axis=0).ravel(),np.diff(err,axis=1).ravel()))))
    cyc=np.angle(np.exp(1j*(np.angle(np.exp(1j*pred))-np.angle(np.exp(1j*wrapped)))))
    return dict(aligned_mae=float(np.mean(np.abs(err))),aligned_rmse=float(np.sqrt(np.mean(err**2))),
                aligned_nrmse=float(np.sqrt(np.mean(err**2))/(target.max()-target.min()+1e-12)),pge=float(pge),
                rewrap_circular_mae=float(np.mean(np.abs(cyc))))


def load_models():
    models={}
    for name in study.METHODS:
        model,_=study.build(name); state=torch.load(RUNS/name/"best.pth",map_location="cpu",weights_only=False)
        model.load_state_dict(state["model"]); model.eval(); models[name]=model
    return models


def main():
    missing=[m for m in study.METHODS if not (RUNS/m/"complete.json").exists()]
    if missing: raise RuntimeError(f"unfinished methods: {missing}")
    results={m:json.loads((RUNS/m/"complete.json").read_text(encoding="utf-8")) for m in study.METHODS}
    histories={m:json.loads((RUNS/m/"history.json").read_text(encoding="utf-8")) for m in study.METHODS}
    models=load_models(); tw=study.load_data("test_input"); tt=study.load_data("test_target")

    rows=[]
    for m in study.METHODS:
        r=results[m]; rows.append(dict(method=LABEL[m],selected_epoch=r["selected_epoch"],epochs=r["epochs"],
            updates=r["total_updates"],parameters=r["parameters"],gpu_hours=r["total_seconds"]/3600,**r["test"]))

    traditional_detail={name:[] for name in TRADITIONAL}; traditional_time={name:0.0 for name in TRADITIONAL}
    wnp,tnp=tw.numpy()[:,0],tt.numpy()[:,0]
    for i,(w,t) in enumerate(zip(wnp,tnp)):
        for name,fn in TRADITIONAL.items():
            begin=time.perf_counter(); p=fn(w); traditional_time[name]+=time.perf_counter()-begin
            traditional_detail[name].append(np_metrics(p,t,w))
        if (i+1)%100==0: print(f"traditional {i+1}/{len(wnp)}",flush=True)
    for name,items in traditional_detail.items():
        metric={k:float(np.mean([x[k] for x in items])) for k in items[0]}
        rows.append(dict(method=name,selected_epoch=0,epochs=0,updates=0,parameters=0,gpu_hours=0.0,
                         latency_ms=traditional_time[name]*1000/len(items),**metric))

    keys=list(rows[0]); keys += [k for k in rows[-1] if k not in keys]
    with (OUT/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    (OUT/"comparison.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")

    fig,axes=plt.subplots(2,1,figsize=(10,8),constrained_layout=True)
    for m,h in histories.items():
        axes[0].plot([x["epoch"] for x in h],[x["train_loss"] for x in h],label=LABEL[m],lw=1)
        axes[1].plot([x["epoch"] for x in h],[x["val_aligned_mae"] for x in h],label=LABEL[m],lw=1)
    axes[0].set(yscale="log",xlabel="Epoch",ylabel="Training objective",title="Training curves")
    axes[1].set(yscale="log",xlabel="Epoch",ylabel="Validation aligned MAE (rad)",title="Validation curves")
    for ax in axes: ax.grid(alpha=.25); ax.legend(ncol=3)
    fig.savefig(FIG/"training_and_validation_curves.png",dpi=200); plt.close(fig)

    names=[x["method"] for x in rows]; vals=[x["aligned_mae"] for x in rows]
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True); ax.bar(names,vals)
    ax.set(ylabel="Test aligned MAE (rad)",title="Full test-set comparison"); ax.tick_params(axis="x",rotation=30)
    ax.grid(axis="y",alpha=.25); fig.savefig(FIG/"test_mae_comparison.png",dpi=200); plt.close(fig)

    chosen=[0,1,2]; deep_predictions={}
    with torch.no_grad():
        for m,model in models.items():
            w=tw[chosen].cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=m!="u3net"): deep_predictions[m]=study.infer(model,m,w).float().cpu().numpy()[:,0]
    qpred={n:[fn(wnp[i]) for i in chosen] for n,fn in TRADITIONAL.items()}
    display=[("Wrapped",wnp[chosen]),("Ground truth",tnp[chosen])]+[(LABEL[m],deep_predictions[m]) for m in study.METHODS]+[(n,np.asarray(qpred[n])) for n in TRADITIONAL]
    fig,axes=plt.subplots(len(display),len(chosen),figsize=(9,2.2*len(display)),constrained_layout=True)
    for r,(name,images) in enumerate(display):
        for c,img in enumerate(images):
            axes[r,c].imshow(img,cmap="viridis"); axes[r,c].axis("off")
            if c==0: axes[r,c].set_title(name,loc="left",fontsize=9)
    fig.savefig(FIG/"qualitative_comparison.png",dpi=180); plt.close(fig)

    lines=["# 非扩散方法完整数据正式对比","", "所有深度方法均从头训练，使用固定的 20,000/500/1,500 训练、验证、测试划分。权重只按验证集对齐 MAE 选择；传统方法在相同 1,500 张测试图上评估。","",
           "| 方法 | 训练轮数 | 最佳轮次 | Updates | 参数量 | 对齐 MAE↓ | 对齐 RMSE↓ | NRMSE↓ | PGE↓ |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(f"| {x['method']} | {x['epochs']} | {x['selected_epoch']} | {x['updates']:,} | {x['parameters']:,} | {x['aligned_mae']:.4f} | {x['aligned_rmse']:.4f} | {x['aligned_nrmse']:.4f} | {x['pge']:.4f} |")
    lines += ["", "U3Net 使用完整 500 轮自恢复加 200 轮蒸馏，并只从蒸馏阶段选择最终权重。Restormer 以 300,000 次更新为停止条件。详细逐轮数据见各方法的 `history.csv`，每轮权重见对应 `weights` 目录。"]
    (OUT/"REPORT.zh-CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__": main()
