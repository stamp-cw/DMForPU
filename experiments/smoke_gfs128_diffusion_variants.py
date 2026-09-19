"""One-sample forward/backward smoke test for each distinct diffusion graph."""
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiments.train_gfs_rme128 as suite


def main():
    for method in ("chen_physics","chen_no_sparse","chen_no_cam","hf_matched","dcc_only","wwfca_only","fdu"):
        suite.seed_all();model,cfg=suite.diffusion_model(method);wrapped=torch.zeros(1,1,128,128,device="cuda")
        target=torch.randn_like(wrapped);sigma=torch.ones(1,device="cuda");timestep=torch.tensor([500],device="cuda")
        noisy=model.scheduler.add_noise(model.normalize(target),torch.randn_like(target),timestep)
        with torch.autocast("cuda",dtype=torch.float16):estimate,phase,stages=model(noisy,wrapped,timestep,sigma);loss=F.mse_loss(estimate,model.normalize(target))
        loss.backward()
        if estimate.shape!=target.shape or not torch.isfinite(loss):raise RuntimeError(method)
        print(method,sum(p.numel() for p in model.parameters()),float(loss.detach()),len(stages),flush=True)
        del model,wrapped,target,sigma,timestep,noisy,estimate,phase,stages,loss;torch.cuda.empty_cache()


if __name__=="__main__":main()
