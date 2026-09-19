"""Global convolutional-high-frequency WWFCA residual downsampler."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import UNet2DModel

from .fdunet_atten import _haar_dwt2


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


class GlobalFrequencyCrossAttention(nn.Module):
    """LL queries convolutionally fused LH/HL/HH over the full feature map."""

    def __init__(self, channels: int, heads: int = 4, max_size: int = 16, dropout: float = 0.05):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels=channels;self.heads=heads;self.head_dim=channels//heads
        self.max_size=max_size;self.dropout=dropout
        self.band_logits=nn.Parameter(torch.zeros(3,channels))
        self.high_in=nn.Conv2d(3*channels,channels,1)
        self.high_norm=nn.GroupNorm(min(32,channels),channels)
        self.high_local=nn.Sequential(nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1,groups=channels),
                                      nn.Conv2d(channels,channels,1))
        self.eta_logit=nn.Parameter(torch.tensor(_logit(0.2)))  # 0.5*sigmoid -> 0.1
        self.query_norm=nn.LayerNorm(channels);self.context_norm=nn.LayerNorm(channels)
        self.to_q=nn.Linear(channels,channels,bias=False)
        self.to_k=nn.Linear(channels,channels,bias=False)
        self.to_v=nn.Linear(channels,channels,bias=False)
        self.output_norm=nn.LayerNorm(channels)
        self.output=nn.Linear(channels,channels)
        # Keep the initial branch visible enough to train. A near-zero output
        # projection combined with the 0.02 residual gate starves q/k/v of
        # gradients (the failure mode observed in v3). Gain 0.1 keeps the
        # initial injected/base RMS ratio in the 1e-4--1e-3 range.
        nn.init.xavier_uniform_(self.output.weight,gain=0.1);nn.init.zeros_(self.output.bias)
        self.relative_bias=nn.Parameter(torch.zeros(heads,2*max_size-1,2*max_size-1))
        nn.init.trunc_normal_(self.relative_bias,std=0.02)
        self.condition_norm=nn.LayerNorm(channels*4+1)
        self.film=nn.Linear(channels*4+1,2*channels)
        nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)

    def _position_bias(self,height:int,width:int,device,dtype):
        if height>self.max_size or width>self.max_size:
            raise ValueError(f"WWFCA-v4 subband {height}x{width} exceeds max_size={self.max_size}")
        yy,xx=torch.meshgrid(torch.arange(height,device=device),torch.arange(width,device=device),indexing="ij")
        coords=torch.stack((yy.flatten(),xx.flatten()))
        relative=coords[:,:,None]-coords[:,None,:]
        iy=relative[0]+self.max_size-1;ix=relative[1]+self.max_size-1
        return self.relative_bias[:,iy,ix].to(dtype=dtype)

    def forward(self,ll,lh,hl,hh,condition:Optional[torch.Tensor]=None):
        batch,channels,height,width=ll.shape
        weights=2.0*torch.sigmoid(self.band_logits)
        bands=[band*weights[index][None,:,None,None] for index,band in enumerate((lh,hl,hh))]
        high=self.high_in(torch.cat(bands,dim=1));eta=0.5*torch.sigmoid(self.eta_logit)
        high=high+eta*self.high_local(self.high_norm(high))
        if condition is not None:
            scale,shift=self.film(self.condition_norm(condition)).chunk(2,dim=1)
            high=high*(1+scale[:,:,None,None])+shift[:,:,None,None]
        query=self.query_norm(ll.flatten(2).transpose(1,2))
        context=self.context_norm(high.flatten(2).transpose(1,2))
        q=self.to_q(query).view(batch,-1,self.heads,self.head_dim).transpose(1,2)
        k=self.to_k(context).view(batch,-1,self.heads,self.head_dim).transpose(1,2)
        v=self.to_v(context).view(batch,-1,self.heads,self.head_dim).transpose(1,2)
        logits=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        logits=logits+self._position_bias(height,width,logits.device,logits.dtype)[None]
        attention=logits.float().softmax(dim=-1).to(logits.dtype)
        attention=F.dropout(attention,p=self.dropout,training=self.training)
        output=torch.matmul(attention,v).transpose(1,2).reshape(batch,height*width,channels)
        output=self.output(self.output_norm(output)).transpose(1,2).reshape(batch,channels,height,width)
        with torch.no_grad():
            entropy=-(attention.float().clamp_min(1e-12)*attention.float().clamp_min(1e-12).log()).sum(-1)
            self.last_attention_entropy=float((entropy/math.log(height*width)).mean())
        return output


class WWFCAv4Downsample(nn.Module):
    """Original HF downsample plus a bounded global WWFCA residual."""

    def __init__(self,base:nn.Module,channels:int,enabled:bool,max_scale:float=0.1):
        super().__init__();self.base=base;self.enabled=enabled;self.max_scale=max_scale
        self.cross=GlobalFrequencyCrossAttention(channels)
        self.gamma_logit=nn.Parameter(torch.tensor(_logit(0.2)))  # max_scale*0.2 = 0.02
        self._condition=None;self._energy_penalty=None
        self._diagnostics={"gamma":0.,"residual_rms_ratio":0.,"attention_entropy":0.}

    def set_condition(self,condition:Optional[torch.Tensor]):self._condition=condition
    def energy_penalty(self):
        return self.gamma_logit.new_zeros(()) if self._energy_penalty is None else self._energy_penalty
    def diagnostics(self):return dict(self._diagnostics)

    def forward(self,x):
        base=self.base(x)
        if not self.enabled:
            self._energy_penalty=base.new_zeros(());self._diagnostics={k:0. for k in self._diagnostics};return base
        ll,lh,hl,hh=_haar_dwt2(x)
        residual=self.cross(ll,lh,hl,hh,self._condition)
        if residual.shape!=base.shape:
            raise ValueError(f"WWFCA residual {tuple(residual.shape)} != downsample {tuple(base.shape)}")
        gamma=self.max_scale*torch.sigmoid(self.gamma_logit)
        injected=gamma*residual
        ratio=injected.float().square().mean().sqrt()/base.float().square().mean().sqrt().clamp_min(1e-8)
        self._energy_penalty=F.relu(ratio-0.01).square()
        self._diagnostics={"gamma":float(gamma.detach()),"residual_rms_ratio":float(ratio.detach()),
                           "attention_entropy":self.cross.last_attention_entropy}
        return base+injected


class WWFCAv4UNet(nn.Module):
    """HF UNet with one parallel WWFCA branch at the 32->16 downsample."""

    def __init__(self,enabled:bool,widths:Tuple[int,...]=(128,128,128,128),layers:int=2):
        super().__init__();self.enabled=enabled
        self.unet=UNet2DModel(sample_size=128,in_channels=2,out_channels=1,layers_per_block=layers,
            block_out_channels=widths,down_block_types=("DownBlock2D",)*len(widths),
            up_block_types=("UpBlock2D",)*len(widths),add_attention=False)
        target=self.unet.down_blocks[-2]
        if target.downsamplers is None or len(target.downsamplers)!=1:
            raise ValueError("Expected one downsampler in the penultimate HF down block")
        target.downsamplers[0]=WWFCAv4Downsample(target.downsamplers[0],widths[-2],enabled)

    @property
    def adapter(self):
        return self.unet.down_blocks[-2].downsamplers[0]

    def load_hf_backbone(self,state:dict):
        mapped={}
        prefix="down_blocks.2.downsamplers.0."
        for key,value in state.items():
            mapped[key.replace(prefix,prefix+"base.",1) if key.startswith(prefix) else key]=value
        result=self.unet.load_state_dict(mapped,strict=False)
        unexpected=list(result.unexpected_keys)
        missing=[key for key in result.missing_keys if ".cross." not in key and not key.endswith("gamma_logit")]
        if unexpected or missing:raise RuntimeError(f"HF weight mapping failed; missing={missing}, unexpected={unexpected}")

    def _time_condition(self,sample,timestep,sigma):
        if not torch.is_tensor(timestep):timestep=torch.tensor([timestep],device=sample.device)
        if timestep.ndim==0:timestep=timestep[None]
        timestep=timestep.to(sample.device).expand(sample.shape[0])
        temb=self.unet.time_embedding(self.unet.time_proj(timestep).to(dtype=sample.dtype))
        noise=torch.log1p(sigma.to(device=sample.device,dtype=temb.dtype).clamp_min(0.0))[:,None]
        return torch.cat((temb,noise),dim=1)

    def energy_penalty(self):return self.adapter.energy_penalty()
    def diagnostics(self):return self.adapter.diagnostics()

    def forward(self,sample,timestep,sigma=None):
        sigma=sample.new_zeros(sample.shape[0]) if sigma is None else sigma
        self.adapter.set_condition(self._time_condition(sample,timestep,sigma))
        try:return self.unet(sample,timestep)
        finally:self.adapter.set_condition(None)
