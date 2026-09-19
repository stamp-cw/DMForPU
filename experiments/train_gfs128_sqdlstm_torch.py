"""Train the native PyTorch SQD-LSTM port on GFS128."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from model.sqdlstm_torch import JointConvSQDLSTMNetTorch, tv_loss_plus_var_loss
from utils.phase_metrics import au_metrics_numpy, u3_aligned_metrics_numpy

OUT = ROOT / "experiments" / "results" / "gfs128_upstream" / "runs" / "sqd_lstm_torch"
SEED, BATCH, EPOCHS, PATIENCE = 42, 4, 100, 10
SNRS = ("clean", 0, 5, 10, 20, 30)
TWO_PI = 2 * math.pi


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); tmp.replace(path)


def seed_all() -> None:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def load_train():
    manifest = json.loads((ROOT / "experiments/results/gfs128_upstream/manifest.json").read_text(encoding="utf-8"))
    with h5py.File(ROOT / "data/GFS128/train.h5", "r") as f:
        wrapped = torch.from_numpy(f["psi"][:])[:, None]
        target = torch.from_numpy(f["phi"][:])[:, None]
    return (wrapped, target, torch.tensor(manifest["train_indices"]), torch.tensor(manifest["validation_indices"]))


def resize(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, (256, 256), mode="bilinear", align_corners=False)


@torch.no_grad()
def validation_loss(model, wrapped, target, ids) -> float:
    model.eval(); total = 0.0; seen = 0
    for off in range(0, len(ids), BATCH):
        chosen = ids[off:off+BATCH]
        x = resize(wrapped[chosen].cuda()); y = resize(target[chosen].cuda())
        loss = tv_loss_plus_var_loss(y, model(x)); total += float(loss) * len(chosen); seen += len(chosen)
    return total / seen


def metrics(pred, target, wrapped):
    raw = pred-target; mean = raw-raw.mean((1,2), keepdims=True); span = np.ptp(target, axis=(1,2))
    integer = raw+np.round(np.median(-raw, (1,2))/TWO_PI)[:,None,None]*TWO_PI
    cycle = np.mod(np.mod(pred+math.pi,TWO_PI)-math.pi-wrapped+math.pi,TWO_PI)-math.pi
    values={"raw_mae":float(np.mean(np.abs(raw))), "integer_aligned_mae":float(np.mean(np.abs(integer))),
            "mean_aligned_mae":float(np.mean(np.abs(mean))), "mean_aligned_rmse":float(np.mean(np.sqrt(np.mean(mean**2,(1,2))))),
            "mean_aligned_nrmse":float(np.mean(np.sqrt(np.mean(mean**2,(1,2)))/np.maximum(span,1e-12))),
            "rewrap_circular_mae":float(np.mean(np.abs(cycle)))}
    values.update({key: float(value.mean()) for key, value in u3_aligned_metrics_numpy(pred, target).items()})
    values.update({key: float(value.mean()) for key, value in au_metrics_numpy(pred, target).items()})
    return values


@torch.no_grad()
def evaluate_file(model, path):
    with h5py.File(path, "r") as f: wrapped=f["psi"][:]; target=f["phi"][:]
    outputs=[]; model.eval()
    for off in range(0,len(wrapped),BATCH):
        x=resize(torch.from_numpy(wrapped[off:off+BATCH,None]).cuda())
        p=model(x); p=F.interpolate(p,(128,128),mode="bilinear",align_corners=False)[:,0]
        outputs.append(p.cpu().numpy())
    return metrics(np.concatenate(outputs),target,wrapped)


def benchmark(steps: int = 30) -> None:
    seed_all(); wrapped,target,train_ids,_=load_train();model=JointConvSQDLSTMNetTorch().cuda()
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    x=resize(wrapped[train_ids[:BATCH]].cuda());y=resize(target[train_ids[:BATCH]].cuda())
    for _ in range(5):
        opt.zero_grad(set_to_none=True);loss=tv_loss_plus_var_loss(y,model(x));loss.backward();opt.step()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True);loss=tv_loss_plus_var_loss(y,model(x));loss.backward();opt.step()
    torch.cuda.synchronize();seconds=(time.perf_counter()-started)/steps
    print(json.dumps({"backend":"native_pytorch","steps":steps,"step_seconds":seconds,"steps_per_second":1/seconds,
                      "trainable_parameters":model.trainable_parameters,"state_parameters":sum(p.numel() for p in model.parameters()),
                      "output_shape":list(model(x).shape),"peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,"loss":float(loss.detach())}))


def train() -> None:
    if (OUT/"complete.json").exists(): print("sqd_lstm_torch complete",flush=True);return
    seed_all();OUT.mkdir(parents=True,exist_ok=True);wrapped,target,train_ids,val_ids=load_train()
    model=JointConvSQDLSTMNetTorch().cuda();opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=1e-3)
    protocol={"method":"sqd_lstm_torch","source_architecture":"third_party/DeepPhaseUnwrap/src/models/architectures.py",
              "port":"native PyTorch, layer-for-layer","input_adapter":"GFS128 bilinear 128->256; prediction 256->128",
              "batch":BATCH,"epochs":EPOCHS,"early_stopping_monitor":"train_loss","patience":PATIENCE,
              "trainable_parameters":model.trainable_parameters,"state_parameters":sum(p.numel() for p in model.parameters())}
    dump(OUT/"protocol.json",protocol);history=[];best=math.inf;bad=0;start=0
    if (OUT/"last.pth").exists():
        state=torch.load(OUT/"last.pth",map_location="cpu",weights_only=False);model.load_state_dict(state["model"]);opt.load_state_dict(state["optimizer"])
        history=state["history"];best=state["best"];bad=state["bad_epochs"];start=state["epoch"]
    for epoch in range(start,EPOCHS):
        model.train();order=train_ids[torch.randperm(len(train_ids),generator=torch.Generator().manual_seed(SEED+epoch))]
        total=0.;seen=0;torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        for off in range(0,len(order),BATCH):
            ids=order[off:off+BATCH];x=resize(wrapped[ids].cuda());y=resize(target[ids].cuda())
            opt.zero_grad(set_to_none=True);prediction=model(x);loss=tv_loss_plus_var_loss(y,prediction)
            if not torch.isfinite(loss):raise RuntimeError(f"nonfinite loss at epoch {epoch+1}")
            loss.backward();opt.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
        train_loss=total/seen;val_loss=validation_loss(model,wrapped,target,val_ids);seconds=time.perf_counter()-begin
        row={"epoch":epoch+1,"train_loss":train_loss,"val_loss":val_loss,"seconds":seconds,"gpu_peak_mib":torch.cuda.max_memory_allocated()/2**20}
        history.append(row);dump(OUT/"history.json",history)
        with (OUT/"history.csv.tmp").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        (OUT/"history.csv.tmp").replace(OUT/"history.csv")
        state={"epoch":epoch+1,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),
               "history":history,"best":best,"bad_epochs":bad,"protocol":protocol}
        if train_loss < best:
            best=train_loss;bad=0;state["best"]=best;state["bad_epochs"]=bad;save(OUT/"best.pth",state)
        else: bad+=1;state["bad_epochs"]=bad
        save(OUT/"last.pth",state);dump(OUT/"status.json",{"state":"training",**row,"epochs":EPOCHS,"best_train_loss":best,"bad_epochs":bad})
        print(f"sqd_lstm_torch {epoch+1}/{EPOCHS} loss={train_loss:.6f} val={val_loss:.6f} {seconds:.1f}s",flush=True)
        if bad>=PATIENCE:break
    best_state=torch.load(OUT/"best.pth",map_location="cpu",weights_only=False);model.load_state_dict(best_state["model"])
    result={"method":"sqd_lstm_torch","epochs_completed":len(history),"selected_epoch":best_state["epoch"],
            "parameters":model.trainable_parameters,"tests":{}}
    for snr in SNRS:
        name="test_clean.h5" if snr=="clean" else f"test_{snr}dB.h5"
        result["tests"][str(snr)]=evaluate_file(model,ROOT/"data/GFS128"/name)
    dump(OUT/"complete.json",result);dump(OUT/"status.json",{"state":"complete",**result})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("benchmark","train"));parser.add_argument("--steps",type=int,default=30)
    args=parser.parse_args();benchmark(args.steps) if args.command=="benchmark" else train()
