"""Retrain and evaluate all requested models on GFS128 and RME128."""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
import experiments.formal_non_diffusion_study as nd
from diffusion.chen_hf_diffusion import ChenHFConfig,ChenHFDiffusion,paired_recorruption,sr_loss,sd_loss
from diffusion.dcc_wwfca_diffusion import DCCWWFCADiffusion
from utils.phase_metrics import au_metrics_torch, u3_aligned_metrics_torch

OUT=ROOT/"experiments"/"results"/"gfs_rme128"
DATASETS=("GFS128",)
CHEN_FLAGS={
    "hf_base":dict(physics=False,sparse=True,adaptive=True,sr=False,sd=False),
    "chen_physics":dict(physics=True,sparse=True,adaptive=True,sr=False,sd=False),
    "chen_sr":dict(physics=False,sparse=True,adaptive=True,sr=True,sd=False),
    "chen_physics_sr":dict(physics=True,sparse=True,adaptive=True,sr=True,sd=False),
    "chen_full":dict(physics=True,sparse=True,adaptive=True,sr=True,sd=True),
    "chen_no_sparse":dict(physics=True,sparse=False,adaptive=True,sr=True,sd=True),
    "chen_no_cam":dict(physics=True,sparse=True,adaptive=False,sr=True,sd=True),
}
MATCHED_DIFFUSION=("hf_matched","dcc_only","wwfca_only","fdu","wwfca_v2_off","wwfca_v2","wwfca_v3_off","wwfca_v3","wwfca_v4_off","wwfca_v4","wwfca_v41_off","wwfca_v41_noattn","wwfca_v41","wwfca_v42_time","wwfca_v42","wwfca_v43")
DIFFUSION_METHODS=tuple(CHEN_FLAGS)+MATCHED_DIFFUSION
METHODS=("dlpu","punet","u3net",*DIFFUSION_METHODS,"sqd_lstm","restormer","uformer")
EPOCHS={"dlpu":100,"punet":300,"u3net":700,"sqd_lstm":100,"restormer":267,"uformer":250,
        **{method:300 for method in DIFFUSION_METHODS}}
EPOCHS.update({"wwfca_v2_off":100,"wwfca_v2":100})
EPOCHS.update({"wwfca_v3_off":50,"wwfca_v3":50})
EPOCHS.update({"wwfca_v4_off":25,"wwfca_v4":25})
EPOCHS.update({"wwfca_v41_off":35,"wwfca_v41":35})
EPOCHS.update({"wwfca_v41_noattn":35})
EPOCHS.update({"wwfca_v42_time":35,"wwfca_v42":35})
EPOCHS.update({"wwfca_v43":35})
BATCH={"dlpu":32,"punet":32,"u3net":10,"sqd_lstm":32,"restormer":4,"uformer":8,
       **{method:8 for method in DIFFUSION_METHODS}}
SEED=42; VAL_COUNT=500; TEST_SNRS=(0,5,10,20,30); TWO_PI=2*math.pi
SELECTION_KEY="val_u3_aligned_nrmse"


def dump(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8"); tmp.replace(path)


def save(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(obj,tmp); tmp.replace(path)


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def prepare_manifest():
    path=OUT/"manifest.json"
    source=json.loads((ROOT/"data"/"DATASETS_V2.json").read_text(encoding="utf-8"))
    source_hash=file_hash(ROOT/"data"/"DATASETS_V2.json")
    if path.exists():
        obj=json.loads(path.read_text(encoding="utf-8"))
        # DATASETS_V2 also tracks unrelated RTS/RBR families. Their later rename
        # changed the global digest without changing these frozen GFS/RME splits.
        for dataset in obj["datasets"].values():
            for relative in (dataset["train"],*dataset["tests"].values()):
                if not (ROOT/relative).exists():raise FileNotFoundError(relative)
        return obj
    order=np.random.default_rng(SEED).permutation(5000); val=sorted(order[:VAL_COUNT].tolist()); train=sorted(order[VAL_COUNT:].tolist())
    obj={"seed":SEED,"source_manifest_sha256":source_hash,"train_indices":train,"validation_indices":val,
         "policy":"fixed 4500/500 split of official-size 5000 training file; five 1000-image test sets untouched",
         "datasets":{d:{"train":str((ROOT/"data"/d/"train.h5").relative_to(ROOT)),
                         "tests":{str(s):str((ROOT/"data"/d/f"test_{s}dB.h5").relative_to(ROOT)) for s in TEST_SNRS}}
                     for d in DATASETS}}
    dump(path,obj); return obj


def read_h5(path):
    with h5py.File(path,"r") as f:
        return (torch.from_numpy(f["psi"][:]),torch.from_numpy(f["phi"][:]),torch.from_numpy(f["snr"][:]))


def data(dataset):
    mf=prepare_manifest(); w,t,s=read_h5(ROOT/mf["datasets"][dataset]["train"])
    tri=torch.tensor(mf["train_indices"]); vai=torch.tensor(mf["validation_indices"])
    tests={snr:read_h5(ROOT/mf["datasets"][dataset]["tests"][str(snr)]) for snr in TEST_SNRS}
    return (w[tri,None],t[tri,None],s[tri]),(w[vai,None],t[vai,None],s[vai]),tests


def wrap(x): return torch.remainder(x+math.pi,TWO_PI)-math.pi


def metric_sums(pred,target,wrapped):
    pred,target,wrapped=pred.float(),target.float(),wrapped.float(); raw=pred-target
    integer_k=torch.round(torch.median((-raw).flatten(1),1).values/TWO_PI)[:,None,None,None]
    integer=raw+integer_k*TWO_PI; mean=raw-raw.flatten(1).mean(1)[:,None,None,None]
    target_range=target.flatten(1).amax(1)-target.flatten(1).amin(1)
    grad=torch.cat(((mean[:,:,1:,:]-mean[:,:,:-1,:]).flatten(1),
                    (mean[:,:,:,1:]-mean[:,:,:,:-1]).flatten(1)),1)
    cycle=wrap(wrap(pred)-wrapped)
    values={"raw_mae":raw.abs().flatten(1).mean(1),"raw_rmse":raw.square().flatten(1).mean(1).sqrt(),
      "integer_aligned_mae":integer.abs().flatten(1).mean(1),"integer_aligned_rmse":integer.square().flatten(1).mean(1).sqrt(),
      "mean_aligned_mae":mean.abs().flatten(1).mean(1),"mean_aligned_rmse":mean.square().flatten(1).mean(1).sqrt(),
      "mean_aligned_nrmse":mean.square().flatten(1).mean(1).sqrt()/target_range.clamp_min(1e-12),
      "pge":grad.abs().mean(1),"rewrap_circular_mae":cycle.abs().flatten(1).mean(1)}
    values.update(u3_aligned_metrics_torch(pred, target))
    values.update(au_metrics_torch(pred, target))
    return {k:float(v.sum().cpu()) for k,v in values.items()}


def u3_inputs(w,std):
    from model.u3net.u3net import grad_op
    return wrap(grad_op(w)),std[:,None],torch.ones_like(w),torch.zeros(*w.shape,2,device=w.device)


def infer(model,method,w,std=None,generator=None):
    if method=="u3net": return model(*u3_inputs(w,std))[0]
    if method in DIFFUSION_METHODS: return model.sample(w,std,generator)
    return model(w)


@torch.no_grad()
def evaluate(model,method,split,batch,seed=9000):
    w,t,s=split; sums={}; n=0; model.eval(); gen=torch.Generator(device="cuda").manual_seed(seed)
    for off in range(0,len(w),batch):
        wi,ti,si=w[off:off+batch].cuda(),t[off:off+batch].cuda(),s[off:off+batch].cuda()
        std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,si/10))
        with torch.autocast("cuda",dtype=torch.float16 if method in DIFFUSION_METHODS else torch.bfloat16,
                            enabled=method!="u3net"):
            p=infer(model,method,wi,std,gen)
        if p.shape!=ti.shape or not torch.isfinite(p).all(): raise RuntimeError(f"invalid {method} prediction")
        for k,v in metric_sums(p,ti,wi).items(): sums[k]=sums.get(k,0.)+v
        n+=len(wi)
    return {k:v/n for k,v in sums.items()}


def per_sample_augment(w,t,gen):
    ow=[]; ot=[]
    for wi,ti in zip(w,t):
        k=int(torch.randint(0,4,(),generator=gen,device=w.device)); wi=torch.rot90(wi,k,(-2,-1)); ti=torch.rot90(ti,k,(-2,-1))
        if bool(torch.randint(0,2,(),generator=gen,device=w.device)):
            dim=-1 if bool(torch.randint(0,2,(),generator=gen,device=w.device)) else -2; wi=torch.flip(wi,(dim,));ti=torch.flip(ti,(dim,))
        ow.append(wi);ot.append(ti)
    return torch.stack(ow),torch.stack(ot)


def u3_loss(model,teacher,w,std,gen):
    from model.u3net.u3net import grad_op
    u=torch.randn(w.shape,device=w.device,generator=gen)*std[:,None,None,None]
    effective=wrap(w+u)-w; plus=wrap(grad_op(w+effective)); cond=std[:,None]
    x0=torch.ones_like(w);a0=torch.zeros(*w.shape,2,device=w.device)
    if teacher is None:
        pred,stages=model(plus,cond,x0,a0); minus=grad_op(w-effective)
        terms=[wrap(minus-grad_op(x)).square().mean()/(len(stages)-i) for i,x in enumerate(stages)]
        return sum(terms),pred
    with torch.no_grad(): target=teacher(plus,cond,x0,a0)[0]
    clean=wrap(grad_op(w)); pred,stages=model(clean,cond,x0,a0)
    terms=[F.l1_loss(grad_op(x),grad_op(target))/(len(stages)-i) for i,x in enumerate(stages)]
    return sum(terms),pred


def non_diff_model(method):
    seed_all(); return nd.build(method)[0]


def state_cpu(model): return {k:v.detach().cpu() for k,v in model.state_dict().items()}


def finite_or_none(value):
    """Keep strict JSON status files valid before a selectable best checkpoint exists."""
    return float(value) if math.isfinite(value) else None


def write_history(folder,h):
    dump(folder/"history.json",h); tmp=folder/"history.csv.tmp"
    with tmp.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=list(h[0]));wr.writeheader();wr.writerows(h)
    tmp.replace(folder/"history.csv")


def train_non_diff(dataset,method):
    folder=OUT/"runs"/dataset/method; folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists(): return
    (train,val,tests)=data(dataset); epochs=EPOCHS[method];batch=BATCH[method]; model=non_diff_model(method)
    optimizer=nd.optimizer_for(method,model); target_updates=300000 if method=="restormer" else None
    protocol={"dataset":dataset,"method":method,"epochs":epochs,"batch":batch,"seed":SEED,"validation_count":VAL_COUNT,
              "selection":"lowest validation U3-aligned NRMSE; U3Net only epochs 501-700",
              "target_updates":target_updates,"manifest_sha256":file_hash(OUT/"manifest.json"),"from_scratch":True}
    dump(folder/"protocol.json",protocol); history=[];best=math.inf;start=0;teacher=None;last=folder/"last.pth"
    if last.exists():
        st=torch.load(last,map_location="cpu",weights_only=False)
        if st["protocol"]!=protocol: raise RuntimeError("resume protocol mismatch")
        model.load_state_dict(st["model"]);optimizer.load_state_dict(st["optimizer"]);history=st["history"];best=st["best"];start=st["epoch"]+1
        if st.get("teacher") is not None: teacher=copy.deepcopy(model).eval().requires_grad_(False);teacher.load_state_dict(st["teacher"])
    total_updates=sum(x["updates"] for x in history)
    for epoch in range(start,epochs):
        if method=="u3net" and epoch==500: teacher=copy.deepcopy(model).eval().requires_grad_(False);optimizer=nd.optimizer_for(method,model)
        lr=nd.lr_for(method,epoch,epochs,teacher is not None)
        for g in optimizer.param_groups:g["lr"]=lr
        model.train();gen=torch.Generator(device="cuda").manual_seed(SEED*100000+epoch);order=torch.randperm(len(train[0]),generator=torch.Generator().manual_seed(SEED+epoch))
        loss_sum=0.;seen=updates=0;begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for off in range(0,len(order),batch):
            if target_updates is not None and total_updates>=target_updates: break
            idx=order[off:off+batch];w,t,s=(x[idx].cuda() for x in train);w,t=per_sample_augment(w,t,gen)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=method!="u3net"):
                if method=="u3net":
                    std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10));loss,p=u3_loss(model,teacher,w,std,gen)
                else: p=model(w);loss,_=nd.supervised_loss(method,p,t)
            if not torch.isfinite(loss) or not torch.isfinite(p).all():raise RuntimeError(f"nonfinite {dataset}/{method}/{epoch+1}")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            n=len(w);loss_sum+=float(loss.detach())*n;seen+=n;updates+=1;total_updates+=1
        val_metrics=evaluate(model,method,val,min(batch,32),seed=10000+epoch);seconds=time.perf_counter()-begin
        row={"epoch":epoch+1,"phase":"distillation" if teacher is not None else "self_recovery" if method=="u3net" else "supervised",
             "lr":lr,"train_loss":loss_sum/seen,"val_mean_aligned_mae":val_metrics["mean_aligned_mae"],"val_mean_aligned_nrmse":val_metrics["mean_aligned_nrmse"],
             "updates":updates,"cumulative_updates":total_updates,"samples":seen,"seconds":seconds,"samples_per_second":seen/seconds,
             "gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        history.append(row);write_history(folder,history);cp={"epoch":epoch,"model":state_cpu(model),"protocol":protocol,"metrics":row}
        save(folder/"weights"/f"epoch_{epoch+1:03d}.pth",cp)
        eligible=method!="u3net" or teacher is not None
        if eligible and row["val_mean_aligned_mae"]<best:best=row["val_mean_aligned_mae"];save(folder/"best.pth",cp)
        save(last,{**cp,"optimizer":optimizer.state_dict(),"teacher":None if teacher is None else state_cpu(teacher),"history":history,"best":best})
        dump(folder/"status.json",{"state":"training","epoch":epoch+1,"epochs":epochs,
                                   "best_val":finite_or_none(best),"updates":total_updates})
        print(f"{dataset} {method} {epoch+1}/{epochs} loss={row['train_loss']:.5f} val={row['val_mean_aligned_mae']:.5f} {seconds:.1f}s",flush=True)
        if target_updates is not None and total_updates>=target_updates:break
    finish(dataset,method,model,folder,tests,history,best)


def diffusion_model(method):
    if method in MATCHED_DIFFUSION:
        model=DCCWWFCADiffusion(method,phase_low=-14*math.pi,phase_high=14*math.pi)
        return model.cuda(),model.cfg
    flags=CHEN_FLAGS[method]
    cfg=ChenHFConfig(phase_low=-14*math.pi,phase_high=14*math.pi,
                     physics=flags["physics"],sparse=flags["sparse"],adaptive=flags["adaptive"])
    return ChenHFDiffusion(cfg).cuda(),cfg


def train_diffusion(dataset,method):
    folder=OUT/"runs"/dataset/method;folder.mkdir(parents=True,exist_ok=True)
    if (folder/"complete.json").exists():return
    train,val,tests=data(dataset);epochs=EPOCHS[method];batch=BATCH[method];seed_all();model,cfg=diffusion_model(method)
    flags=CHEN_FLAGS.get(method,dict(sr=False,sd=False));is_v3=method in ("wwfca_v3_off","wwfca_v3");is_v4=method in ("wwfca_v4_off","wwfca_v4");is_v41=method in ("wwfca_v41_off","wwfca_v41_noattn","wwfca_v41");is_v42=method in ("wwfca_v42_time","wwfca_v42");is_v43=method=="wwfca_v43";is_adapter=is_v4 or is_v41 or is_v42 or is_v43
    last=folder/"last.pth";pretrained_path=OUT/"runs"/dataset/"hf_matched"/"weights"/"epoch_299.pth"
    if is_adapter and not last.exists():
        pretrained=torch.load(pretrained_path,map_location="cpu",weights_only=False)["model"]
        source={key[len("backbone."):]:value for key,value in pretrained.items() if key.startswith("backbone.")}
        model.backbone.load_hf_backbone(source)
    weight_decay=1e-4 if method in ("wwfca_v2_off","wwfca_v2","wwfca_v3_off","wwfca_v3","wwfca_v4_off","wwfca_v4","wwfca_v41_off","wwfca_v41_noattn","wwfca_v41","wwfca_v42_time","wwfca_v42","wwfca_v43") else 0
    if is_adapter:
        prefix="backbone.unet.down_blocks.2.downsamplers.0."
        branch=[p for name,p in model.named_parameters() if name.startswith(prefix) and ".base." not in name]
        branch_ids={id(p) for p in branch};main=[p for p in model.parameters() if id(p) not in branch_ids]
        opt=torch.optim.AdamW([{"params":main,"lr":0.,"role":"main"},
                               {"params":branch,"lr":1e-4,"role":"branch"}],weight_decay=weight_decay)
    elif is_v3:
        branch_names=("mid_block.frequency","mid_block.gate","mid_block.condition_norm")
        branch=[p for name,p in model.named_parameters() if any(token in name for token in branch_names)]
        branch_ids={id(p) for p in branch};main=[p for p in model.parameters() if id(p) not in branch_ids]
        opt=torch.optim.AdamW([{"params":main,"lr":2e-4,"lr_scale":1.0},
                               {"params":branch,"lr":2e-5,"lr_scale":0.1}],weight_decay=weight_decay)
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=weight_decay)
    scaler=torch.amp.GradScaler("cuda");teacher=None;phase=210
    protocol={"dataset":dataset,"method":method,"epochs":epochs,"batch":batch,"seed":SEED,"distill_start":phase if flags["sd"] else None,
      "validation":"full fixed 500-image validation split every epoch; all unified metrics","selection":"lowest validation U3-aligned NRMSE (U3Net min-max range alignment)","checkpointing":"model weights every epoch; resumable optimizer/scaler in last.pth","manifest_sha256":file_hash(OUT/"manifest.json"),"config":model.config_dict(),"from_scratch":not is_adapter}
    if is_v3:
        protocol["v3_training"]={"epochs":50,"base_lr":2e-4,"branch_lr_ratio":0.1,"weight_decay":1e-4,
                                  "gate_regularization_weight":0.01,"gate_off_epochs":10,
                                  "gate_ramp_epochs":20,"validation_seed":20000}
    if is_v4:
        protocol["v4_training"]={"pretrained_checkpoint":str(pretrained_path.relative_to(ROOT)),"pretrained_sha256":file_hash(pretrained_path),
                                  "adapter_only_epochs":5,"joint_epochs":20,"adapter_lr_first":1e-4,
                                  "adapter_lr_joint":5e-5,"backbone_lr_joint":1e-5,"energy_limit":0.01,
                                  "weight_decay":1e-4,"validation_seed":20000}
    if is_v41:
        protocol["v41_training"]={"pretrained_checkpoint":str(pretrained_path.relative_to(ROOT)),"pretrained_sha256":file_hash(pretrained_path),
                                   "adapter_only_epochs":15,"joint_epochs":20,"adapter_lr_first":1e-4,
                                   "adapter_lr_joint":5e-5,"backbone_lr_joint":1e-6,
                                   "loss":"unchanged HF clean-x0 MSE; energy ratio monitored but not optimized",
                                   "attention":"disabled; fixed G=0.5" if method=="wwfca_v41_noattn" else ("enabled" if method=="wwfca_v41" else "branch disabled"),
                                   "energy_limit_monitor":0.01,"weight_decay":1e-4,"validation_seed":20000}
    if is_v42:
        protocol["v42_training"]={"pretrained_checkpoint":str(pretrained_path.relative_to(ROOT)),"pretrained_sha256":file_hash(pretrained_path),
                                   "adapter_only_epochs":15,"joint_epochs":20,"adapter_lr_first":1e-4,
                                   "adapter_lr_joint":5e-5,"backbone_lr_joint":1e-6,
                                   "loss":"unchanged HF clean-x0 MSE; energy ratio monitored but not optimized",
                                   "gate":"gamma(t,sigma)" if method=="wwfca_v42" else "gamma(t)",
                                   "weight_decay":1e-4,"validation_seed":20000}
    if is_v43:
        protocol["v43_training"]={"pretrained_checkpoint":str(pretrained_path.relative_to(ROOT)),"pretrained_sha256":file_hash(pretrained_path),
                                   "adapter_only_epochs":15,"joint_epochs":20,"adapter_lr_first":1e-4,
                                   "adapter_lr_joint":5e-5,"backbone_lr_joint":1e-6,
                                   "loss":"unchanged HF clean-x0 MSE; energy ratio monitored but not optimized",
                                   "attention_order":"H queries X for validation, then X queries validated H for retrieval",
                                   "residual":"validated H local skip plus scaled retrieval, then learned-gamma injection",
                                   "weight_decay":1e-4,"validation_seed":20000}
    dump(folder/"protocol.json",protocol);history=[];best=math.inf;start=0
    if last.exists():
        st=torch.load(last,map_location="cpu",weights_only=False)
        stored_protocol=dict(st["protocol"]);stored_protocol.pop("selection",None)
        expected_protocol=dict(protocol);expected_protocol.pop("selection",None)
        if stored_protocol!=expected_protocol:raise RuntimeError("resume protocol mismatch")
        model.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"]);scaler.load_state_dict(st["scaler"]);history=st["history"];best=st["best"];start=st["epoch"]+1
        if st.get("teacher") is not None:teacher=copy.deepcopy(model).eval().requires_grad_(False);teacher.load_state_dict(st["teacher"])
        eligible=[row for row in history if SELECTION_KEY in row and (not flags["sd"] or row.get("phase")=="distillation")]
        if eligible:
            selected=min(eligible,key=lambda row:row[SELECTION_KEY]);best=selected[SELECTION_KEY]
            selected_state=torch.load(folder/"weights"/f"epoch_{selected['epoch']:03d}.pth",map_location="cpu",weights_only=False)
            selected_state["protocol"]=protocol;save(folder/"best.pth",selected_state)
    for epoch in range(start,epochs):
        if flags["sd"] and epoch>=phase and teacher is None:teacher=copy.deepcopy(model).eval().requires_grad_(False)
        model.train();order=torch.randperm(len(train[0]),generator=torch.Generator().manual_seed(SEED+epoch));gen=torch.Generator(device="cuda").manual_seed(SEED*100000+epoch)
        lr=2e-4*min(1.0,(epoch+1)/10) if method in ("wwfca_v2_off","wwfca_v2","wwfca_v3_off","wwfca_v3") else 2e-4
        if is_adapter:
            for group in opt.param_groups:
                freeze_epochs=15 if (is_v41 or is_v42 or is_v43) else 5
                main_lr=1e-6 if (is_v41 or is_v42 or is_v43) else 1e-5
                group["lr"]=(0. if epoch<freeze_epochs else main_lr) if group["role"]=="main" else (1e-4 if epoch<freeze_epochs else 5e-5)
            lr=max(group["lr"] for group in opt.param_groups)
        else:
            for group in opt.param_groups:group["lr"]=lr*group.get("lr_scale",1.0)
        gate_scale=0.0 if epoch<10 else min(1.0,(epoch-9)/20.0)
        if is_v3:model.backbone.set_gate_scale(gate_scale)
        ts=torch.randint(cfg.train_steps,(len(train[0]),),generator=gen,device="cuda");dn=torch.randn(train[1].shape,generator=gen,device="cuda");rn=torch.randn(train[0].shape,generator=gen,device="cuda")
        sums={"total":0.,"supervised":0.,"sr":0.,"sd":0.}
        if is_v3:sums["gate_reg"]=0.
        if is_adapter:sums["energy_reg"]=0.
        gate_diag={"gate_mean_abs":0.,"gate_max_abs":0.,"gate_saturated_fraction":0.,"frequency_residual_rms_ratio":0.}
        adapter_diag={"gamma":0.,"residual_rms_ratio":0.,"attention_entropy":0.}
        if is_v41:adapter_diag.update({"gate_spatial_std":0.,"temperature":0.,"distance_strength":0.})
        if is_v42:adapter_diag.update({"gamma_min":0.,"gamma_max":0.,"time_strength":0.,"noise_strength":0.,
                                      "gate_spatial_std":0.,"temperature":0.,"distance_strength":0.})
        if is_v43:adapter_diag.update({"validation_entropy":0.,"retrieval_entropy":0.,"validation_gate_mean":0.,
                                      "validation_gate_spatial_std":0.,"validation_scale":0.,"retrieval_scale":0.})
        seen=updates=0;begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for off in range(0,len(order),batch):
            idx=order[off:off+batch];w,t,s=(x[idx].cuda() for x in train);std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10));x0=model.normalize(t);xt=model.scheduler.add_noise(x0,dn[idx],ts[idx]);opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                estimate,phi,_=model(xt,w,ts[idx],std);supervised=(estimate-x0).square().mean();sr=supervised.new_zeros(());sd=supervised.new_zeros(())
                if flags["sr"] or teacher is not None:
                    plus,negative=paired_recorruption(w,std,rn[idx]);_,_,stages=model(xt,plus,ts[idx],std);weights=[1/(len(stages)-j) for j in range(len(stages))]
                    if flags["sr"]:sr=sum(sr_loss(p,negative)*q for p,q in zip(stages,weights))/sum(weights)
                    if teacher is not None:
                        with torch.no_grad():_,pseudo,_=teacher(xt,plus,ts[idx],std)
                        sd=sd_loss(phi,pseudo)
                gate_reg=model.backbone.gate_regularization() if is_v3 else supervised.new_zeros(())
                energy_reg=model.backbone.energy_penalty() if is_adapter else supervised.new_zeros(())
                # v4.1 is a structure-only ablation: its optimization objective
                # must remain the same clean-x0 MSE as HF. The energy value is
                # logged as a diagnostic and is not added to the v4.1 loss.
                loss=supervised+.05*sr+.05*sd+(.01*gate_reg if is_v3 else 0.)+(energy_reg if is_v4 else 0.)
            if not torch.isfinite(loss):raise RuntimeError(f"nonfinite {dataset}/{method}/{epoch+1}")
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);before=scaler.get_scale();scaler.step(opt);scaler.update();updates+=int(scaler.get_scale()>=before)
            n=len(w);seen+=n
            components={"total":loss,"supervised":supervised,"sr":sr,"sd":sd}
            if is_v3:components["gate_reg"]=gate_reg
            if is_adapter:components["energy_reg"]=energy_reg
            for k,v in components.items():sums[k]+=float(v.detach())*n
            if is_v3:
                diagnostics=model.backbone.gate_diagnostics()
                for k,v in diagnostics.items():
                    gate_diag[k]=max(gate_diag[k],v) if k=="gate_max_abs" else gate_diag[k]+v*n
            if is_adapter:
                diagnostics=model.backbone.diagnostics()
                for k,v in diagnostics.items():adapter_diag[k]+=v*n
        validation_seed=20000 if method in ("wwfca_v2_off","wwfca_v2","wwfca_v3_off","wwfca_v3","wwfca_v4_off","wwfca_v4","wwfca_v41_off","wwfca_v41_noattn","wwfca_v41","wwfca_v42_time","wwfca_v42","wwfca_v43") else 20000+epoch
        vm=evaluate(model,method,val,batch,seed=validation_seed);seconds=time.perf_counter()-begin
        row={"epoch":epoch+1,"phase":"distillation" if teacher is not None else "diffusion","lr":lr,**{f"train_{k}":v/seen for k,v in sums.items()},
             **{f"val_{key}":value for key,value in vm.items()},"updates":updates,"samples":seen,"seconds":seconds,"samples_per_second":seen/seconds,"gpu_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
        if is_v3:
            row["gate_scale"]=gate_scale
            row.update({k:(v if k=="gate_max_abs" else v/seen) for k,v in gate_diag.items()})
        if is_adapter:
            freeze_epochs=15 if (is_v41 or is_v42 or is_v43) else 5
            row["phase"]="adapter_only" if epoch<freeze_epochs else "joint_finetune"
            row.update({k:v/seen for k,v in adapter_diag.items()})
        history.append(row);write_history(folder,history);cp={"epoch":epoch,"model":state_cpu(model),"protocol":protocol,"metrics":row};save(folder/"weights"/f"epoch_{epoch+1:03d}.pth",cp)
        eligible=not flags["sd"] or teacher is not None
        if eligible and row[SELECTION_KEY]<best:best=row[SELECTION_KEY];save(folder/"best.pth",cp)
        save(last,{**cp,"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"teacher":None if teacher is None else state_cpu(teacher),"history":history,"best":best})
        dump(folder/"status.json",{"state":"training","epoch":epoch+1,"epochs":epochs,
                                   "best_val":finite_or_none(best)})
        print(f"{dataset} {method} {epoch+1}/{epochs} loss={row['train_total']:.5f} val_u3_nrmse={row[SELECTION_KEY]:.6f} {seconds:.1f}s",flush=True)
    finish(dataset,method,model,folder,tests,history,best)


def finish(dataset,method,model,folder,tests,history,best):
    st=torch.load(folder/"best.pth",map_location="cpu",weights_only=False);model.load_state_dict(st["model"])
    result={"dataset":dataset,"method":method,"selected_epoch":st["epoch"]+1,"selection_metric":"u3_aligned_nrmse","best_validation_u3_aligned_nrmse":best,
            "epochs_completed":len(history),"parameters":sum(p.numel() for p in model.parameters()),"tests":{},"total_seconds":sum(x["seconds"] for x in history)}
    for snr in TEST_SNRS:result["tests"][str(snr)]=evaluate(model,method,(tests[snr][0][:,None],tests[snr][1][:,None],tests[snr][2]),BATCH[method],seed=30000+snr)
    clean_path=ROOT/"data"/dataset/"test_clean.h5"
    if clean_path.exists():
        clean=read_h5(clean_path)
        result["tests"]["clean"]=evaluate(model,method,(clean[0][:,None],clean[1][:,None],clean[2]),BATCH[method],seed=31000)
    dump(folder/"complete.json",result);dump(folder/"status.json",{"state":"complete",**result})


def train(dataset,method):
    if method in DIFFUSION_METHODS:train_diffusion(dataset,method)
    else:train_non_diff(dataset,method)


def smoke():
    prepare_manifest();actual=data("GFS128")[0]
    for method in METHODS:
        seed_all();w=actual[0][:2].cuda();t=actual[1][:2].cuda();s=actual[2][:2].cuda()
        std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10))
        if method in DIFFUSION_METHODS:
            m,cfg=diffusion_model(method);ts=torch.randint(cfg.train_steps,(2,),device="cuda");x0=m.normalize(t);xt=m.scheduler.add_noise(x0,torch.randn_like(x0),ts);p,_,_=m(xt,w,ts,std);loss=F.mse_loss(p,x0)
        else:
            m=non_diff_model(method)
            if method=="u3net":loss,p=u3_loss(m,None,w,std,torch.Generator(device="cuda").manual_seed(1))
            else:p=m(w);loss,_=nd.supervised_loss(method,p,t)
        loss.backward();print(method,sum(p.numel() for p in m.parameters()),float(loss.detach()),flush=True);del m,w,t,p,loss;torch.cuda.empty_cache()


def queue():
    prepare_manifest()
    for dataset in DATASETS:
        for method in METHODS:
            log=OUT/f"{dataset}_{method}.log";err=OUT/f"{dataset}_{method}.err.log"
            with log.open("a",encoding="utf-8") as o,err.open("a",encoding="utf-8") as e:
                code=subprocess.call([sys.executable,"-B","-u",str(Path(__file__)),"train","--dataset",dataset,"--method",method],cwd=ROOT,stdout=o,stderr=e)
            if code:raise SystemExit(f"failed {dataset}/{method}: {err}")
    subprocess.check_call([sys.executable,"-B",str(ROOT/"experiments"/"analyze_gfs_rme128.py")],cwd=ROOT)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("smoke","train","queue"));ap.add_argument("--dataset",choices=DATASETS);ap.add_argument("--method",choices=METHODS);a=ap.parse_args()
    if a.command=="smoke":smoke()
    elif a.command=="train":train(a.dataset,a.method)
    else:queue()
