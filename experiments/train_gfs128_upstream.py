"""Train GFS128 baselines by importing the unmodified official repositories.

The only project-owned layer here is the HDF5 adapter, split bookkeeping, neutral
metrics, and checkpoint logging. Network classes come directly from third_party/.
"""
from __future__ import annotations

import argparse, contextlib, copy, csv, hashlib, importlib.util, io, json, math, random, subprocess, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.phase_metrics import au_metrics_torch, u3_aligned_metrics_torch
THIRD = ROOT / "third_party"
OUT = ROOT / "experiments" / "results" / "gfs128_upstream"
METHODS = ("u3net", "dlpu", "punet", "uformer", "restormer", "vurnet")
EPOCHS = {"u3net": 700, "dlpu": 100, "punet": 300, "uformer": 250, "restormer": 267,
          "vurnet": 500}
BATCH = {"u3net": 10, "dlpu": 32, "punet": 16, "uformer": 4, "restormer": 4,
         "vurnet": 20}
COMMITS = {
    "u3net": "47c0af31fee5ee7f442aa433a5fb3483a4376334",
    "dlpu": "3c84f34846dd8e9fbbdc613d161e278c2279d1bc",
    "punet": "5be3c0117e20742bf4f68dc828fc0399d1e362b4",
    "uformer": "65fc970a8ffc09605faca74ed016ee93c9ad8a36",
    "restormer": "68dc6ac472db26f16361150cb7a96a1bc87da93f",
    "vurnet": "59168a529fe879daeac8fa2f03ea60b91ea4e0b3",
}
REPOS = {
    "u3net": "Unsupervised-PU", "dlpu": "Phase_unwrapping_by_U-Net",
    "punet": "Deformation-Monitoring-Dev", "uformer": "Uformer", "restormer": "Restormer",
    "vurnet": "VUR-Net",
}
SNRS = (0, 5, 10, 20, 30)
SEED = 42
TWO_PI = 2 * math.pi
SELECTION_KEY = "val_u3_aligned_nrmse"


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); tmp.replace(path)


def module_from(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def source_file(method: str) -> Path:
    return {
        "u3net": THIRD/"Unsupervised-PU"/"modules"/"network.py",
        "dlpu": THIRD/"Phase_unwrapping_by_U-Net"/"Network.py",
        "punet": THIRD/"Deformation-Monitoring-Dev"/"model"/"PUNet.py",
        "uformer": THIRD/"Uformer"/"model.py",
        "restormer": THIRD/"Restormer"/"basicsr"/"models"/"archs"/"restormer_arch.py",
        "vurnet": THIRD/"VUR-Net"/"VURNet.py",
    }[method]


def seed_all(seed: int = SEED) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build(method: str) -> torch.nn.Module:
    if method == "u3net":
        repo = THIRD/"Unsupervised-PU"
        sys.path.insert(0, str(repo))
        try:
            from modules.network import network
            model = network(3)
        finally: sys.path.pop(0)
        return model.cuda()
    if method == "dlpu":
        return module_from("official_dlpu_network", source_file(method)).UNet().cuda()
    if method == "punet":
        model = module_from("official_punet_network", source_file(method)).PUNet(num_channels=1)
        utils = module_from("official_punet_utils", THIRD/"Deformation-Monitoring-Dev"/"utils"/"utils.py")
        utils.init_weight(model, torch.nn.init.kaiming_normal_, torch.nn.BatchNorm2d, 1e-3, .1, mode="fan_in")
        return model.cuda()
    if method == "uformer":
        cls = module_from("official_uformer_model", source_file(method)).Uformer
        return cls(img_size=128, in_chans=3, dd_in=3, embed_dim=32, win_size=8,
                   token_projection="linear", token_mlp="leff",
                   depths=[1,2,8,8,2,8,8,2,1], modulator=True).cuda()
    if method == "vurnet":
        return module_from("official_vurnet_network", source_file(method)).VURnet().cuda()
    cls = module_from("official_restormer_arch", source_file(method)).Restormer
    return cls(inp_channels=1, out_channels=1, dim=48, num_blocks=[4,6,6,8],
               num_refinement_blocks=4, heads=[1,2,4,8], ffn_expansion_factor=2.66,
               bias=False, LayerNorm_type="BiasFree", dual_pixel_task=False).cuda()


def manifest() -> dict:
    source = ROOT/"experiments"/"results"/"gfs_rme128"/"manifest.json"
    if source.exists():
        old = json.loads(source.read_text(encoding="utf-8"))
        train, val = old["train_indices"], old["validation_indices"]
    else:
        order = np.random.default_rng(SEED).permutation(5000)
        val, train = sorted(order[:500].tolist()), sorted(order[500:].tolist())
    result = {"dataset":"GFS128", "train_indices":train, "validation_indices":val,
              "policy":"same fixed 4500/500 split as prior GFS128 comparison",
              "official_sources":{m:{"repo":REPOS[m],"commit":COMMITS[m],"source":str(source_file(m).relative_to(ROOT)),"sha256":sha256(source_file(m))} for m in METHODS}}
    dump(OUT/"manifest.json", result); return result


def load_data():
    mf = manifest()
    with h5py.File(ROOT/"data"/"GFS128"/"train.h5", "r") as f:
        w=torch.from_numpy(f["psi"][:])[:,None]; t=torch.from_numpy(f["phi"][:])[:,None]; s=torch.from_numpy(f["snr"][:])
    tri=torch.tensor(mf["train_indices"]); vai=torch.tensor(mf["validation_indices"])
    tests={}
    for snr in SNRS:
        with h5py.File(ROOT/"data"/"GFS128"/f"test_{snr}dB.h5","r") as f:
            tests[snr]=(torch.from_numpy(f["psi"][:])[:,None],torch.from_numpy(f["phi"][:])[:,None],torch.from_numpy(f["snr"][:]))
    return (w[tri],t[tri],s[tri]),(w[vai],t[vai],s[vai]),tests


def wrap(x): return torch.remainder(x + math.pi, TWO_PI) - math.pi


def grad_op(x):
    result=torch.zeros(*x.shape,2,device=x.device,dtype=x.dtype)
    result[:,:,:,1:,0]=x[...,1:]-x[...,:-1]
    result[:,:,1:,:,1]=x[:,:,1:,:]-x[:,:,:-1,:]
    return result


def neutral_metrics(pred, target, wrapped):
    raw=pred.float()-target.float(); mean=raw-raw.flatten(1).mean(1)[:,None,None,None]
    integer=raw+torch.round(torch.median((-raw).flatten(1),1).values/TWO_PI)[:,None,None,None]*TWO_PI
    span=target.flatten(1).amax(1)-target.flatten(1).amin(1)
    grad=torch.cat(((mean[:,:,1:]-mean[:,:,:-1]).flatten(1),(mean[:,:,:,1:]-mean[:,:,:,:-1]).flatten(1)),1)
    cycle=wrap(wrap(pred)-wrapped)
    vals={"raw_mae":raw.abs().flatten(1).mean(1),"raw_rmse":raw.square().flatten(1).mean(1).sqrt(),
          "integer_aligned_mae":integer.abs().flatten(1).mean(1),"integer_aligned_rmse":integer.square().flatten(1).mean(1).sqrt(),
          "mean_aligned_mae":mean.abs().flatten(1).mean(1),
          "mean_aligned_rmse":mean.square().flatten(1).mean(1).sqrt(),"mean_aligned_nrmse":mean.square().flatten(1).mean(1).sqrt()/span.clamp_min(1e-12),
          "pge":grad.abs().mean(1),"rewrap_circular_mae":cycle.abs().flatten(1).mean(1)}
    vals.update(u3_aligned_metrics_torch(pred, target))
    vals.update(au_metrics_torch(pred, target))
    return {k:float(v.sum().cpu()) for k,v in vals.items()}


def official_scaled_nrmse(pred, target):
    pmin=pred.flatten(1).amin(1)[:,None,None,None]; pmax=pred.flatten(1).amax(1)[:,None,None,None]
    tmin=target.flatten(1).amin(1)[:,None,None,None]; tmax=target.flatten(1).amax(1)[:,None,None,None]
    scaled=(pred-pmin)/(pmax-pmin).clamp_min(1e-12)*(tmax-tmin)+tmin
    return ((scaled-target).square().flatten(1).mean(1).sqrt()/(tmax-tmin).flatten()*100).sum().item()


def model_input(method, w): return w.repeat(1,3,1,1) if method=="uformer" else w
def prediction(method, out): return out.mean(1,keepdim=True) if method=="uformer" else out


def forward_model(method, model, value):
    """Run upstream code while suppressing VUR-Net's per-batch debug print."""
    if method == "vurnet":
        with contextlib.redirect_stdout(io.StringIO()):
            return model(value)
    return model(value)


def u3_forward(model,w,std):
    official_grad=sys.modules["modules.network"].grad_op
    x=torch.ones_like(w); a=torch.zeros(*w.shape,2,device=w.device)
    return model(wrap(official_grad(w)),std[:,None],x,a)[0]


@torch.no_grad()
def evaluate(model,method,split,batch):
    model.eval(); w,t,s=split; sums={}; n=0
    for off in range(0,len(w),batch):
        wi,ti,si=w[off:off+batch].cuda(),t[off:off+batch].cuda(),s[off:off+batch].cuda()
        if method=="u3net":
            std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,si/10)); p=u3_forward(model,wi,std)
        else: p=prediction(method,forward_model(method,model,model_input(method,wi)))
        for k,v in neutral_metrics(p,ti,wi).items(): sums[k]=sums.get(k,0.)+v
        if method=="u3net": sums["official_scaled_nrmse_percent"]=sums.get("official_scaled_nrmse_percent",0.)+official_scaled_nrmse(p,ti)
        n+=len(wi)
    return {k:v/n for k,v in sums.items()}


def augment(w,t,gen):
    k=int(torch.randint(0,4,(),generator=gen,device=w.device)); w=torch.rot90(w,k,(-2,-1));t=torch.rot90(t,k,(-2,-1))
    if bool(torch.randint(0,2,(),generator=gen,device=w.device)):
        dim=-1 if bool(torch.randint(0,2,(),generator=gen,device=w.device)) else -2; w=torch.flip(w,(dim,));t=torch.flip(t,(dim,))
    return w,t


def u3_loss(model,teacher,w,std,gen):
    official_grad=sys.modules["modules.network"].grad_op
    official_metrics=sys.modules.get("official_u3_metrics") or module_from("official_u3_metrics",THIRD/"Unsupervised-PU"/"metrics.py")
    noise=torch.randn(w.shape,device=w.device,generator=gen)*std[:,None,None,None]
    effective=wrap(w+noise)-w; plus=wrap(official_grad(w+effective)); x=torch.ones_like(w);a=torch.zeros(*w.shape,2,device=w.device)
    if teacher is None:
        pred,stages=model(plus,std[:,None],x,a); minus=official_grad(w-effective)
        return sum(official_metrics.Loss_SR(minus,stage)/(len(stages)-i) for i,stage in enumerate(stages)),pred
    with torch.no_grad(): target=teacher(plus,std[:,None],x,a)[0]
    pred,stages=model(wrap(official_grad(w)),std[:,None],x,a)
    return sum(official_metrics.Loss_SD(target,stage)/(len(stages)-i) for i,stage in enumerate(stages)),pred


def optimizer(method,model):
    if method=="dlpu": return torch.optim.Adam(model.parameters(),lr=.01)
    if method=="punet": return torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    if method=="u3net": return torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8)
    if method=="uformer": return torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=.02)
    if method=="vurnet": return torch.optim.Adam(model.parameters(),lr=1e-4)
    return torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4)


def lr_value(method,epoch,update,total_updates):
    if method=="punet": return 1e-3*max(0.,1-update/total_updates)**.95
    if method=="uformer": return 2e-4*(.5**(epoch//50))
    if method=="restormer":
        if update<92000:return 3e-4
        q=min(1.,(update-92000)/208000);return 1e-6+.5*(3e-4-1e-6)*(1+math.cos(math.pi*q))
    if method=="u3net": return 1e-3*.99**(epoch-500 if epoch>=500 else epoch)
    if method=="vurnet": return 1e-4
    return .01


def train(method, epochs_override=None):
    folder=OUT/"runs"/method; folder.mkdir(parents=True,exist_ok=True)
    target_epochs = int(epochs_override) if epochs_override is not None else EPOCHS[method]
    if target_epochs < 1: raise ValueError("epochs must be positive")
    if (folder/"complete.json").exists() and epochs_override is None:
        print(method,"complete");return
    seed_all(); train_set,val,tests=load_data(); model=build(method); opt=optimizer(method,model); teacher=None
    epochs,batch=target_epochs,BATCH[method]; updates_per_epoch=math.ceil(len(train_set[0])/batch); total_target=300000 if method=="restormer" else epochs*updates_per_epoch
    protocol={"method":method,"official_repo":REPOS[method],"official_commit":COMMITS[method],"source_sha256":sha256(source_file(method)),"epochs":epochs,"batch":batch,"target_updates":total_target,
              "selection":"lowest validation U3-aligned NRMSE (U3Net min-max range alignment)",
              "adapter":"GFS128 H5 + fixed 4500/500 split + neutral metrics", "source_code_modified":False,
              "uformer_channels":"official 3-channel Uformer-B; GFS channel repeated and predictions averaged" if method=="uformer" else None,
              "vurnet_protocol":"README successful-run setting: Adam 1e-4, pixel L1, 500 epochs; upstream network unchanged" if method=="vurnet" else None}
    dump(folder/"protocol.json",protocol); history=[];best=math.inf;start=0;global_update=0;last=folder/"last.pth"
    if last.exists():
        st=torch.load(last,map_location="cpu",weights_only=False); model.load_state_dict(st["model"]);opt.load_state_dict(st["optimizer"]);history=st["history"];best=st["best"];start=st["epoch"]+1;global_update=st["global_update"]
        if st.get("teacher") is not None: teacher=copy.deepcopy(model).eval().requires_grad_(False);teacher.load_state_dict(st["teacher"])
        # Rebuild the selector when resuming runs created before the selection
        # metric changed from mean-aligned MAE to the U3Net-style NRMSE.
        eligible=[row for row in history if SELECTION_KEY in row and (method!="u3net" or row.get("phase")=="distillation")]
        if eligible:
            selected=min(eligible,key=lambda row:row[SELECTION_KEY]);best=selected[SELECTION_KEY]
            selected_state=torch.load(folder/"weights"/f"epoch_{selected['epoch']:03d}.pth",map_location="cpu",weights_only=False)
            selected_state["protocol"]=protocol;save(folder/"best.pth",selected_state)
    for epoch in range(start,epochs):
        if method=="u3net" and epoch==500: teacher=copy.deepcopy(model).eval().requires_grad_(False);opt=optimizer(method,model)
        model.train();order=torch.randperm(len(train_set[0]),generator=torch.Generator().manual_seed(SEED+epoch));gen=torch.Generator(device="cuda").manual_seed(SEED*100000+epoch)
        total=seen=0.;begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        for off in range(0,len(order),batch):
            if global_update>=total_target:break
            ids=order[off:off+batch];w,t,s=(x[ids].cuda() for x in train_set)
            if method in ("u3net","punet","uformer","restormer"):w,t=augment(w,t,gen)
            lr=lr_value(method,epoch,global_update,total_target)
            for group in opt.param_groups:group["lr"]=lr
            opt.zero_grad(set_to_none=True)
            if method=="u3net":
                std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10));loss,p=u3_loss(model,teacher,w,std,gen)
            else:
                out=forward_model(method,model,model_input(method,w)); target=model_input(method,t)
                if method=="dlpu":loss=F.l1_loss(out,target)
                elif method=="punet":loss=F.mse_loss(out,target)
                elif method=="uformer":loss=torch.sqrt((out-target).square()+1e-6).mean()
                else:loss=F.l1_loss(out,target)
                p=prediction(method,out)
            if not torch.isfinite(loss) or not torch.isfinite(p).all():raise RuntimeError(f"nonfinite {method} epoch {epoch+1}")
            loss.backward()
            if method=="restormer":torch.nn.utils.clip_grad_norm_(model.parameters(),.01)
            opt.step();n=len(w);total+=float(loss.detach())*n;seen+=n;global_update+=1
        vm=evaluate(model,method,val,min(batch,16));seconds=time.perf_counter()-begin
        row={"epoch":epoch+1,"phase":"distillation" if teacher is not None else "self_recovery" if method=="u3net" else "supervised","lr":lr,"train_loss":total/seen,"updates":global_update,"seconds":seconds,"samples_per_second":seen/seconds,"gpu_peak_mib":torch.cuda.max_memory_allocated()/2**20,
             **{f"val_{key}":value for key,value in vm.items()}}
        history.append(row);dump(folder/"history.json",history)
        with (folder/"history.csv.tmp").open("w",newline="",encoding="utf-8") as f:wr=csv.DictWriter(f,fieldnames=list(row));wr.writeheader();wr.writerows(history)
        (folder/"history.csv.tmp").replace(folder/"history.csv")
        state={"epoch":epoch,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"metrics":row,"protocol":protocol}
        if row[SELECTION_KEY]<best:best=row[SELECTION_KEY];save(folder/"best.pth",state)
        # Keep a model-only checkpoint for every epoch so learning curves and
        # post-hoc checkpoint selection can be reproduced exactly.
        save(folder/"weights"/f"epoch_{epoch+1:03d}.pth",state)
        save(last,{**state,"optimizer":opt.state_dict(),"teacher":None if teacher is None else {k:v.detach().cpu() for k,v in teacher.state_dict().items()},"history":history,"best":best,"global_update":global_update})
        dump(folder/"status.json",{"state":"training","epoch":epoch+1,"epochs":epochs,"updates":global_update,"best_val":best})
        print(f"{method} {epoch+1}/{epochs} loss={row['train_loss']:.6f} val_u3_nrmse={row[SELECTION_KEY]:.6f} {seconds:.1f}s",flush=True)
        if global_update>=total_target:break
    # Official U3Net reports its final epoch; neutral comparison also keeps validation-best.
    final_state=torch.load(last,map_location="cpu",weights_only=False); model.load_state_dict(final_state["model"])
    result={"method":method,"official_repo":REPOS[method],"official_commit":COMMITS[method],"epochs_completed":len(history),"updates":global_update,"parameters":sum(p.numel() for p in model.parameters()),"tests":{}}
    for snr in SNRS:result["tests"][str(snr)]=evaluate(model,method,tests[snr],batch)
    dump(folder/"complete.json",result);dump(folder/"status.json",{"state":"complete",**result})


def smoke():
    train_set,_,_=load_data()
    for method in METHODS:
        seed_all();model=build(method);w,t,s=(x[:1].cuda() for x in train_set);opt=optimizer(method,model);opt.zero_grad()
        if method=="u3net":
            std=torch.sqrt(torch.tensor(10**.1,device="cuda")/torch.pow(10.,s/10));loss,p=u3_loss(model,None,w,std,torch.Generator(device="cuda").manual_seed(1))
        else:
            out=forward_model(method,model,model_input(method,w));target=model_input(method,t);p=prediction(method,out)
            loss=F.mse_loss(out,target) if method=="punet" else F.l1_loss(out,target)
        loss.backward();print(method,sum(x.numel() for x in model.parameters()),tuple(p.shape),float(loss.detach()),flush=True)
        del model,w,t,p,loss,opt;torch.cuda.empty_cache()


def queue():
    for method in METHODS:
        log=OUT/f"{method}.log";err=OUT/f"{method}.err.log";log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("a",encoding="utf-8") as o,err.open("a",encoding="utf-8") as e:
            code=subprocess.call([sys.executable,"-B","-u",str(Path(__file__)),"train","--method",method],cwd=ROOT,stdout=o,stderr=e)
        if code:raise SystemExit(f"{method} failed; see {err}")
    sqd_log=OUT/"sqd_lstm.log";sqd_err=OUT/"sqd_lstm.err.log"
    with sqd_log.open("a",encoding="utf-8") as o,sqd_err.open("a",encoding="utf-8") as e:
        code=subprocess.call([sys.executable,"-B","-u",str(ROOT/"experiments"/"train_gfs128_sqdlstm_upstream.py")],cwd=ROOT,stdout=o,stderr=e)
    if code:raise SystemExit(f"sqd_lstm failed; see {sqd_err}")
    subprocess.check_call([sys.executable,"-B",str(ROOT/"experiments"/"analyze_gfs128_upstream.py")],cwd=ROOT)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("command",choices=("smoke","train","queue"));ap.add_argument("--method",choices=METHODS);ap.add_argument("--epochs",type=int,default=None);args=ap.parse_args()
    if args.command=="smoke":smoke()
    elif args.command=="train":train(args.method,args.epochs)
    else:queue()
