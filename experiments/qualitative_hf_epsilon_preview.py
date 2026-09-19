from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiments.train_gfs128_t200_prediction_study import RUNS,SELECTION_KEY,build_model,read_test
RUN=RUNS/'hf_epsilon'; OUT=RUN/'current_best_epoch_preview'; OUT.mkdir(parents=True,exist_ok=True)
# Keep the requested comparison checkpoint fixed at epoch 280.
e=280
m=build_model('hf_epsilon').eval(); s=torch.load(RUN/'weights'/f'epoch_{e:03d}.pth',map_location='cpu',weights_only=False); m.load_state_dict(s['model'])
fig,ax=plt.subplots(6,4,figsize=(12,16),squeeze=False)
for ri,c in enumerate(('clean','0','5','10','20','30')):
 w,t,snr=read_test(c); wi,ti,si=w[:1].cuda(),t[:1].cuda(),snr[:1].cuda(); sig=torch.sqrt(torch.tensor(10**.1,device='cuda')/torch.pow(10.,si/10))
 with torch.autocast('cuda',dtype=torch.float16): p=m.sample(wi,sig,generator=torch.Generator(device='cuda').manual_seed(66000+ri),steps=25)
 imgs=(wi[0,0].float().cpu().numpy(),p[0,0].float().cpu().numpy(),ti[0,0].float().cpu().numpy(),np.abs(p[0,0].float().cpu().numpy()-ti[0,0].float().cpu().numpy()))
 for ci,img in enumerate(imgs): ax[ri,ci].imshow(img,cmap='magma' if ci==3 else 'viridis'); ax[ri,ci].axis('off');
 ax[ri,0].set_ylabel(f'{c} dB');
for ci,title in enumerate(('wrapped','prediction','target','abs error')): ax[0,ci].set_title(title)
fig.suptitle(f'HF epsilon qualitative preview, epoch {e}, 25 steps'); fig.tight_layout(); fig.savefig(OUT/'qualitative_preview.png',dpi=180); plt.close(fig)
print('saved',OUT/'qualitative_preview.png','epoch',e)
