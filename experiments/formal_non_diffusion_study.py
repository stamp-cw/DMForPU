"""Full-data, method-specific training study for all non-diffusion baselines."""
from __future__ import annotations

import argparse, copy, csv, importlib, json, math, random, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import experiments.full_200_study as common
from utils.util import dict2namespace

OUT = ROOT / "experiments" / "results" / "formal_non_diffusion"
CACHE_OUT = ROOT / "experiments" / "results" / "full_200"
METHODS = ("dlpu", "sqd_lstm", "punet", "uformer", "restormer", "u3net")
SPECS = {
    "dlpu": dict(epochs=100, batch=32, config="dlpu_synpu_128_big.yaml", module="model.unet.dlpu", loss="l1"),
    "sqd_lstm": dict(epochs=100, batch=32, config="sqd_lstm_synpu_128_big.yaml", module="model.lstm.sqd_lstm", loss="sqd"),
    "punet": dict(epochs=300, batch=32, config="punet_synpu_128_big.yaml", module="model.unet.punet", loss="mse"),
    "uformer": dict(epochs=250, batch=8, config="uformer_synpu_128_big.yaml", module="model.transformer.uformer", loss="charbonnier"),
    # 20,000 / 4 = 5,000 updates per epoch; 60 epochs = exactly 300k updates.
    "restormer": dict(epochs=60, batch=4, config="restormer_synpu_128_big.yaml", module="model.transformer.restormer", loss="l1"),
    "u3net": dict(epochs=700, batch=10, config="u3net_synpu_128_big.yaml", module="model.u3net.u3net", loss="u3"),
}
SEED = 42
SNR_LEVELS = (0.0, 5.0, 10.0, 20.0, 30.0)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_torch(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); tmp.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(kind: str) -> torch.Tensor:
    return torch.from_numpy(np.array(np.load(CACHE_OUT / "cache" / f"{kind}.npy", mmap_mode="r"), copy=True))


def build(method: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    spec = SPECS[method]
    cfg = yaml.safe_load((ROOT / "configs" / spec["config"]).read_text(encoding="utf-8"))
    cfg["mode"] = "train"
    cfg.setdefault("logger", None)
    importlib.import_module(spec["module"])
    from selector.model_selector import _MODELS
    model = _MODELS[cfg["model"]["name"]](dict2namespace(cfg)).cuda()
    if method == "punet":
        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, mode="fan_in")
                if layer.bias is not None: torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.normal_(layer.weight, 1.0, 1e-3); torch.nn.init.constant_(layer.bias, 0.1)
    return model, cfg


def optimizer_for(method: str, model: torch.nn.Module):
    if method == "punet": return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if method in ("uformer", "restormer"):
        return torch.optim.AdamW(model.parameters(), lr=2e-4 if method == "uformer" else 1e-4,
                                 betas=(0.9, .999), eps=1e-8, weight_decay=1e-4)
    lr = 1e-3 if method == "u3net" else 1e-4
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-8)


def lr_for(method: str, epoch: int, total: int, distill: bool = False) -> float:
    if method == "punet": return 1e-3 * max(0.0, 1 - epoch / total) ** .95
    if method == "uformer":
        if epoch < 3: return 2e-4 * (epoch + 1) / 3
        return 1e-6 + .5 * (2e-4 - 1e-6) * (1 + math.cos(math.pi * (epoch - 3) / (total - 3)))
    if method == "restormer": return 1e-6 + .5 * (1e-4 - 1e-6) * (1 + math.cos(math.pi * epoch / total))
    if method == "u3net": return 1e-3 * .99 ** (epoch - 500 if distill else epoch)
    return 1e-4


def augment_pair(w: torch.Tensor, t: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # One deterministic dihedral transform per batch; no interpolation changes phase values.
    k = int(torch.randint(0, 4, (), generator=gen, device=w.device))
    w, t = torch.rot90(w, k, (-2, -1)), torch.rot90(t, k, (-2, -1))
    if bool(torch.randint(0, 2, (), generator=gen, device=w.device)):
        dim = -1 if bool(torch.randint(0, 2, (), generator=gen, device=w.device)) else -2
        w, t = torch.flip(w, (dim,)), torch.flip(t, (dim,))
    return w, t


def u3_inputs(w: torch.Tensor, std: torch.Tensor):
    from model.u3net.u3net import grad_op
    return common.wrap(grad_op(w)), std, torch.ones_like(w), torch.zeros(*w.shape, 2, device=w.device)


def u3_step(model, teacher, wrapped, gen):
    from model.u3net.u3net import grad_op
    ids = torch.randint(0, len(SNR_LEVELS), (len(wrapped),), generator=gen, device=wrapped.device)
    snr = torch.tensor(SNR_LEVELS, device=wrapped.device)[ids]
    std = torch.sqrt(torch.tensor(10 ** .1, device=wrapped.device) / torch.pow(10.0, snr / 10))[:, None]
    noise = torch.randn(wrapped.shape, generator=gen, device=wrapped.device) * std[:, :, None, None]
    noisy = common.wrap(wrapped + noise)
    plus, cond, x0, a0 = u3_inputs(noisy, std)
    if teacher is None:
        pred, stages = model(plus, cond, x0, a0)
        minus = grad_op(2 * wrapped - noisy)
        terms = [common.wrap(minus - grad_op(x)).square().mean() / (len(stages) - i) for i, x in enumerate(stages)]
        return sum(terms), pred, {"self_recovery": float(sum(terms).detach()), "distillation": 0.0}
    with torch.no_grad(): target = teacher(plus, cond, x0, a0)[0]
    clean, cond, x0, a0 = u3_inputs(wrapped, std)
    pred, stages = model(clean, cond, x0, a0)
    terms = [F.l1_loss(grad_op(x), grad_op(target)) / (len(stages) - i) for i, x in enumerate(stages)]
    return sum(terms), pred, {"self_recovery": 0.0, "distillation": float(sum(terms).detach())}


def supervised_loss(method: str, p: torch.Tensor, t: torch.Tensor):
    if method == "sqd_lstm": return common.sqd_loss(p, t)
    if method == "mse" or SPECS[method]["loss"] == "mse": return F.mse_loss(p, t), {}
    if SPECS[method]["loss"] == "charbonnier": return torch.sqrt((p - t).square() + 1e-6).mean(), {}
    return F.l1_loss(p, t), {}


def infer(model, method: str, w: torch.Tensor):
    if method != "u3net": return model(w)
    std = torch.full((len(w), 1), math.sqrt(10 ** .1 / 1000), device=w.device)
    return model(*u3_inputs(w, std))[0]


@torch.no_grad()
def evaluate(model, method, ws, ts, batch):
    model.eval(); sums = {}; count = 0
    for off in range(0, len(ws), batch):
        w, t = ws[off:off+batch].cuda(), ts[off:off+batch].cuda()
        ctx = torch.autocast("cuda", dtype=torch.bfloat16, enabled=method != "u3net")
        with ctx: p = infer(model, method, w)
        if p.shape != t.shape or not torch.isfinite(p).all(): raise RuntimeError(f"invalid prediction: {method}")
        for key, value in common.metric_sums(p, t, w).items(): sums[key] = sums.get(key, 0.0) + value
        count += len(w)
    return {k: v / count for k, v in sums.items()}


def write_history(folder: Path, history):
    atomic_json(folder / "history.json", history)
    tmp = folder / "history.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    tmp.replace(folder / "history.csv")


def train(method: str):
    spec = SPECS[method]; epochs, batch = spec["epochs"], spec["batch"]
    folder = OUT / "runs" / method; folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists(): print(f"{method}: complete"); return
    seed_all(SEED); model, cfg = build(method); optimizer = optimizer_for(method, model)
    protocol = dict(method=method, epochs=epochs, batch_size=batch, seed=SEED, loss=spec["loss"],
                    selection="lowest validation aligned MAE; test unseen until completion",
                    precision="FP32 for U3Net; BF16 autocast with FP32 parameters for other methods",
                    augmentation="dihedral for PUNet/Uformer/U3Net",
                    snr="U3Net per-sample uniform choice from [0,5,10,20,30] dB; validation 30 dB",
                    target_updates=300000 if method == "restormer" else None, config=cfg)
    atomic_json(folder / "protocol.json", protocol)
    teacher = None; history = []; best = math.inf; start = 0
    last = folder / "last.pth"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["protocol"] != protocol: raise RuntimeError(f"protocol mismatch for {method}")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        history, best, start = state["history"], state["best"], state["epoch"] + 1
        if state.get("teacher") is not None:
            teacher = copy.deepcopy(model).eval().requires_grad_(False); teacher.load_state_dict(state["teacher"])
    train_w, train_t = load_data("train_input"), load_data("train_target")
    val_w, val_t = load_data("val_input"), load_data("val_target")
    test_w, test_t = load_data("test_input"), load_data("test_target")
    parameters = sum(p.numel() for p in model.parameters()); total_start = time.perf_counter()
    for epoch in range(start, epochs):
        if method == "u3net" and epoch == 500:
            teacher = copy.deepcopy(model).eval().requires_grad_(False); optimizer = optimizer_for(method, model)
        lr = lr_for(method, epoch, epochs, teacher is not None)
        for group in optimizer.param_groups: group["lr"] = lr
        model.train(); torch.cuda.reset_peak_memory_stats(); begin = time.perf_counter()
        order = torch.randperm(len(train_w), generator=torch.Generator().manual_seed(SEED + epoch))
        gen = torch.Generator(device="cuda").manual_seed(SEED * 100000 + epoch)
        totals = dict(loss=0.0, variance=0.0, tv=0.0, self_recovery=0.0, distillation=0.0); updates = seen = 0
        for off in range(0, len(order), batch):
            idx = order[off:off+batch]; w, t = train_w[idx].cuda(), train_t[idx].cuda()
            if method in ("punet", "uformer", "u3net"): w, t = augment_pair(w, t, gen)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=method != "u3net"):
                if method == "u3net": loss, pred, parts = u3_step(model, teacher, w, gen)
                else: pred = model(w); loss, parts = supervised_loss(method, pred, t)
            if not torch.isfinite(loss) or not torch.isfinite(pred).all(): raise RuntimeError(f"non-finite {method} epoch {epoch+1}")
            loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad): raise RuntimeError(f"non-finite gradient {method} epoch {epoch+1}")
            optimizer.step(); n = len(w); seen += n; updates += 1; totals["loss"] += float(loss.detach()) * n
            for key, value in parts.items(): totals[key] += value * n
        train_seconds = time.perf_counter() - begin
        val = evaluate(model, method, val_w, val_t, min(batch, 32))
        row = dict(epoch=epoch+1, phase="distillation" if teacher is not None else "self_recovery" if method=="u3net" else "supervised",
                   lr=lr, train_loss=totals["loss"]/seen, train_self_recovery=totals["self_recovery"]/seen,
                   train_variance=totals["variance"]/seen, train_tv=totals["tv"]/seen,
                   train_distillation=totals["distillation"]/seen, val_aligned_mae=val["aligned_mae"],
                   val_aligned_rmse=val["aligned_rmse"], val_aligned_nrmse=val["aligned_nrmse"], val_pge=val["pge"],
                   val_rewrap_circular_mae=val["rewrap_circular_mae"], updates=updates, cumulative_updates=sum(x["updates"] for x in history)+updates,
                   samples=seen, train_seconds=train_seconds, epoch_seconds=time.perf_counter()-begin,
                   samples_per_second=seen/train_seconds, gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        history.append(row); write_history(folder, history)
        model_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        checkpoint = dict(epoch=epoch, model=model_state, protocol=protocol, metrics=row)
        atomic_torch(folder / "weights" / f"epoch_{epoch+1:03d}.pth", checkpoint)
        if val["aligned_mae"] < best: best = val["aligned_mae"]; atomic_torch(folder / "best.pth", checkpoint)
        state = dict(checkpoint, optimizer=optimizer.state_dict(), history=history, best=best,
                     teacher=None if teacher is None else {k:v.detach().cpu() for k,v in teacher.state_dict().items()})
        atomic_torch(last, state); atomic_json(folder / "status.json", dict(state="training", epoch=epoch+1, epochs=epochs, best_val=best))
        print(f"{method} epoch {epoch+1:03d}/{epochs} loss={row['train_loss']:.6f} val={val['aligned_mae']:.6f} time={row['epoch_seconds']:.1f}s", flush=True)
    best_state = torch.load(folder / "best.pth", map_location="cpu", weights_only=False); model.load_state_dict(best_state["model"])
    test = evaluate(model, method, test_w, test_t, min(batch, 32))
    result = dict(method=method, selected_epoch=best_state["epoch"]+1, parameters=parameters, test=test,
                  epochs=epochs, total_updates=sum(x["updates"] for x in history), total_seconds=sum(x["epoch_seconds"] for x in history),
                  completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    atomic_json(folder / "complete.json", result); atomic_json(folder / "status.json", dict(state="complete", **result))


def smoke():
    for method in METHODS:
        seed_all(SEED); model, _ = build(method); b = min(2, SPECS[method]["batch"])
        w = torch.randn(b,1,128,128,device="cuda"); t = torch.randn_like(w); gen=torch.Generator(device="cuda").manual_seed(42)
        if method == "u3net": loss,p,_=u3_step(model,None,w,gen)
        else: p=model(w); loss,_=supervised_loss(method,p,t)
        loss.backward(); print(method, sum(x.numel() for x in model.parameters()), tuple(p.shape), float(loss), flush=True)
        del model,w,t,p,loss; torch.cuda.empty_cache()


def queue():
    OUT.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        log=OUT/f"{method}.log"; err=OUT/f"{method}.err.log"
        with log.open("a",encoding="utf-8") as o, err.open("a",encoding="utf-8") as e:
            code=subprocess.call([sys.executable,"-B","-u",str(Path(__file__)),"train","--method",method],stdout=o,stderr=e,cwd=ROOT)
        if code: raise SystemExit(f"{method} failed with {code}; see {err}")
    subprocess.check_call([sys.executable,"-B",str(ROOT/"experiments"/"analyze_formal_non_diffusion.py")],cwd=ROOT)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("smoke","train","queue")); ap.add_argument("--method",choices=METHODS)
    args=ap.parse_args()
    if args.command=="smoke": smoke()
    elif args.command=="train": train(args.method)
    else: queue()
