"""Chen-inspired gradient physics around an unmodified Hugging Face UNet.

This is a supervised diffusion adaptation, not a reproduction of U3Net.
No FDU model, attention block, wavelet, or checkpoint is used.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler


def wrap(x):
    return torch.atan2(x.sin(),x.cos())


def gradient(x):
    return torch.stack((F.pad(x[...,1:]-x[...,:-1],(1,0,0,0)),
                        F.pad(x[...,1:,:]-x[...,:-1,:],(0,0,1,0))),dim=-1)


def adjoint(g):
    gx=g[...,1:,0];gy=g[...,1:,:,1]
    px=F.pad(gx,(1,1,0,0));py=F.pad(gy,(0,0,1,1))
    return px[...,:-1]-px[...,1:]+py[...,:-1,:]-py[...,1:,:]


def paired_recorruption(w,sigma,noise):
    """Mirror official data.py: U = wrap(Y + sigma*z) - Y."""
    plus=wrap(w+sigma[:,None,None,None]*noise)
    effective_u=plus-w
    return plus,gradient(w-effective_u)


def sr_loss(pred,target_gradient):
    return wrap(gradient(pred.float())-target_gradient.float()).square().mean()/math.pi**2


def sd_loss(student,teacher):
    # Official code metrics.py uses gradient L1, unlike paper Eq.14's squared L2.
    return F.l1_loss(gradient(student.float()),gradient(teacher.detach().float()))/math.pi


@dataclass
class ChenHFConfig:
    image_size:int=128
    channels:tuple=(32,64,64)
    phase_low:float=0.
    phase_high:float=14*math.pi
    train_steps:int=1000
    inference_steps:int=5
    physics:bool=True
    sparse:bool=True
    adaptive:bool=True
    stages:int=3
    inner_steps:int=3


class GradientUnroll(nn.Module):
    """Bounded data-gradient iterations, sparse residuals and noise-aware steps.

The HF x0 estimate provides the phase prior; this is not U3Net's SubNN_X.
The data correction has zero spatial mean and cannot infer an absolute offset.
"""
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        if cfg.adaptive:
            self.cam=nn.Sequential(nn.Linear(1,32),nn.SiLU(),nn.Linear(32,cfg.stages*3))
            nn.init.zeros_(self.cam[-1].weight);nn.init.zeros_(self.cam[-1].bias)
        else:self.parameters_by_stage=nn.Parameter(torch.zeros(cfg.stages*3))
        if cfg.sparse:
            self.error_net=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.PReLU(),nn.Conv2d(16,2,3,padding=1))
            nn.init.zeros_(self.error_net[-1].weight);nn.init.zeros_(self.error_net[-1].bias)

    def forward(self,prior,w,sigma):
        prior=prior.float();w=w.float();sigma=sigma.float();g=wrap(gradient(w));x=prior
        values=self.cam(torch.log1p(sigma[:,None])) if self.cfg.adaptive else self.parameters_by_stage[None].expand(len(w),-1)
        values=values.reshape(len(w),3,self.cfg.stages).sigmoid();outputs=[]
        for stage in range(self.cfg.stages):
            step=.2*values[:,0,stage,None,None,None]
            mix=values[:,1,stage,None,None,None]
            threshold=.02+values[:,2,stage,None,None,None,None]
            if self.cfg.sparse:
                b=g-gradient(x)
                residual=b[:,0].permute(0,3,1,2)
                correction=self.error_net(residual).float().permute(0,2,3,1)[:,None]
                b=b+correction
                e=b.sign()*F.relu(b.abs()-threshold)
            else:e=torch.zeros_like(g)
            v=x;previous=x;momentum=1.
            for _ in range(self.cfg.inner_steps):
                nxt=v-step*adjoint(gradient(v)-g+e)
                next_momentum=(1+math.sqrt(1+4*momentum**2))/2
                v=nxt+(momentum-1)/next_momentum*(nxt-previous)
                previous=nxt;momentum=next_momentum
            x=mix*previous+(1-mix)*prior
            outputs.append(x)
        return x,outputs


class ChenHFDiffusion(nn.Module):
    def __init__(self,cfg=None):
        super().__init__();self.cfg=cfg or ChenHFConfig();cfg=self.cfg
        # Direct HF class, never a local FDUNet subclass.
        self.unet=UNet2DConditionModel(sample_size=cfg.image_size,in_channels=2,out_channels=1,
            layers_per_block=1,block_out_channels=tuple(cfg.channels),norm_num_groups=8,
            cross_attention_dim=32,attention_head_dim=8,
            down_block_types=('DownBlock2D','DownBlock2D','CrossAttnDownBlock2D'),
            up_block_types=('CrossAttnUpBlock2D','UpBlock2D','UpBlock2D'))
        self.physics=GradientUnroll(cfg) if cfg.physics else None
        self.scheduler=DDPMScheduler(num_train_timesteps=cfg.train_steps,prediction_type='sample',clip_sample=False)

    def normalize(self,phi):return 2*(phi-self.cfg.phase_low)/(self.cfg.phase_high-self.cfg.phase_low)-1
    def denormalize(self,x):return self.cfg.phase_low+(x+1)*(self.cfg.phase_high-self.cfg.phase_low)/2

    def forward(self,noisy,w,t,sigma):
        hidden=torch.zeros(len(w),1,32,device=w.device,dtype=w.dtype)
        raw=self.unet(torch.cat((noisy,w/math.pi),1),t,encoder_hidden_states=hidden).sample.float()
        phi=self.denormalize(raw)
        if self.physics:phi,stages=self.physics(phi,w,sigma)
        else:stages=[phi]
        return self.normalize(phi),phi,stages

    @torch.no_grad()
    def sample(self,w,sigma,generator=None,steps=None):
        scheduler=DDPMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(steps or self.cfg.inference_steps,device=w.device)
        x=torch.randn(w.shape,device=w.device,dtype=w.dtype,generator=generator)
        for t in scheduler.timesteps:
            pred,_,_=self(x,w,t,sigma)
            x=scheduler.step(pred,t,x,generator=generator).prev_sample
        return self.denormalize(x)

    def config_dict(self):return asdict(self.cfg)
