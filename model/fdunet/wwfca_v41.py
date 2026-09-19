"""WWFCA-v4.1: global LL-guided modulation with a spatial high-frequency path."""
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


class GlobalFrequencyGatedAttention(nn.Module):
    """Use global LL<-H attention to modulate, rather than replace, local H."""

    def __init__(self, channels: int, condition_dim: int, heads: int = 4,
                 max_size: int = 16, dropout: float = 0.05,
                 attention_enabled: bool = True):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels; self.heads = heads; self.head_dim = channels // heads
        self.attention_enabled = attention_enabled
        self.max_size = max_size; self.dropout = dropout
        # Keep the three directional bands separate until after spatial filtering.
        self.band_convs = nn.ModuleList([
            nn.Sequential(nn.GroupNorm(min(32, channels), channels), nn.SiLU(),
                          nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                          nn.Conv2d(channels, channels, 1)) for _ in range(3)
        ])
        self.high_fuse = nn.Conv2d(3 * channels, channels, 1)
        self.high_norm = nn.GroupNorm(min(32, channels), channels)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.film = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

        self.query_norm = nn.LayerNorm(channels); self.context_norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_size - 1, 2 * max_size - 1))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        # temperature = 1 + 15*sigmoid(logit); distance strength = 4*sigmoid(logit)
        self.temperature_logit = nn.Parameter(torch.tensor(_logit(3.0 / 15.0)))  # 4.0
        self.distance_logit = nn.Parameter(torch.tensor(_logit(1.0 / 4.0)))      # 1.0
        self.gate_norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1); nn.init.zeros_(self.gate.bias)
        self.alpha_logit = nn.Parameter(torch.tensor(_logit(0.2)))  # 0.5*sigmoid -> 0.1
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1); nn.init.zeros_(self.output.bias)
        self.last_diagnostics = {"attention_entropy": 0.0, "gate_spatial_std": 0.0,
                                 "temperature": 4.0, "distance_strength": 1.0}

    def _geometry(self, height: int, width: int, device, dtype):
        if height > self.max_size or width > self.max_size:
            raise ValueError(f"WWFCA-v4.1 subband {height}x{width} exceeds max_size={self.max_size}")
        yy, xx = torch.meshgrid(torch.arange(height, device=device),
                                torch.arange(width, device=device), indexing="ij")
        coords = torch.stack((yy.flatten(), xx.flatten()))
        relative = coords[:, :, None] - coords[:, None, :]
        iy = relative[0] + self.max_size - 1; ix = relative[1] + self.max_size - 1
        bias = self.relative_bias[:, iy, ix].to(dtype=dtype)
        dy = relative[0].float() / max(height - 1, 1)
        dx = relative[1].float() / max(width - 1, 1)
        distance = torch.sqrt(dy.square() + dx.square()).to(dtype=dtype)
        return bias, distance

    def forward(self, ll, lh, hl, hh, condition: Optional[torch.Tensor] = None):
        batch, channels, height, width = ll.shape
        directional = [conv(band) for conv, band in zip(self.band_convs, (lh, hl, hh))]
        high = self.high_fuse(torch.cat(directional, dim=1))
        if condition is not None:
            scale, shift = self.film(self.condition_norm(condition)).chunk(2, dim=1)
            high = high * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        temperature = 1 + 15 * torch.sigmoid(self.temperature_logit)
        distance_strength = 4 * torch.sigmoid(self.distance_logit)
        if self.attention_enabled:
            query = self.query_norm(ll.flatten(2).transpose(1, 2))
            context = self.context_norm(high.flatten(2).transpose(1, 2))
            q = self.to_q(query).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            k = self.to_k(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            v = self.to_v(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
            q = F.normalize(q.float(), dim=-1).to(query.dtype)
            k = F.normalize(k.float(), dim=-1).to(context.dtype)
            relative_bias, distance = self._geometry(height, width, q.device, q.dtype)
            logits = temperature * torch.matmul(q, k.transpose(-2, -1))
            logits = logits + relative_bias[None] - distance_strength * distance[None, None]
            attention = logits.float().softmax(dim=-1).to(logits.dtype)
            attention = F.dropout(attention, p=self.dropout, training=self.training)
            attended = torch.matmul(attention, v).transpose(1, 2).reshape(batch, height * width, channels)
            gate = torch.sigmoid(self.gate(self.gate_norm(attended)))
            gate_map = gate.transpose(1, 2).reshape(batch, channels, height, width)
            entropy = -(attention.float().clamp_min(1e-12) *
                        attention.float().clamp_min(1e-12).log()).sum(-1)
            entropy_value = float((entropy / math.log(height * width)).mean())
            spatial_std = float(gate_map.float().std(dim=(-2, -1)).mean())
        else:
            # Exact directional high-frequency residual: no LL query, Q/K/V,
            # attention matrix, or learned gate participates in the forward.
            gate_map = high.new_full((batch, channels, height, width), 0.5)
            entropy_value = 0.0; spatial_std = 0.0
        alpha = 0.5 * torch.sigmoid(self.alpha_logit)
        # The identity high-frequency path prevents global attention from
        # averaging away spatial detail. Attention only selects/modulates it.
        modulated = high * (1 + alpha * (2 * gate_map - 1))
        output = self.output(self.high_norm(modulated))
        with torch.no_grad():
            self.last_diagnostics = {
                "attention_entropy": entropy_value,
                "gate_spatial_std": spatial_std,
                "temperature": float(temperature.detach()),
                "distance_strength": float(distance_strength.detach()),
            }
        return output


class WWFCAv41Downsample(nn.Module):
    def __init__(self, base: nn.Module, channels: int, condition_dim: int,
                 enabled: bool, max_scale: float = 0.1,
                 attention_enabled: bool = True):
        super().__init__(); self.base = base; self.enabled = enabled; self.max_scale = max_scale
        self.cross = GlobalFrequencyGatedAttention(
            channels, condition_dim, attention_enabled=attention_enabled)
        self.gamma_logit = nn.Parameter(torch.tensor(_logit(0.2)))  # gamma=0.02
        self._condition = None; self._energy_penalty = None
        self._diagnostics = {"gamma": 0.0, "residual_rms_ratio": 0.0,
                             **self.cross.last_diagnostics}

    def set_condition(self, condition): self._condition = condition
    def energy_penalty(self):
        return self.gamma_logit.new_zeros(()) if self._energy_penalty is None else self._energy_penalty
    def diagnostics(self): return dict(self._diagnostics)

    def forward(self, x):
        base = self.base(x)
        if not self.enabled:
            self._energy_penalty = base.new_zeros(())
            self._diagnostics = {key: 0.0 for key in self._diagnostics}
            return base
        ll, lh, hl, hh = _haar_dwt2(x)
        residual = self.cross(ll, lh, hl, hh, self._condition)
        gamma = self.max_scale * torch.sigmoid(self.gamma_logit)
        injected = gamma * residual
        ratio = (injected.float().square().mean().sqrt() /
                 base.float().square().mean().sqrt().clamp_min(1e-8))
        self._energy_penalty = F.relu(ratio - 0.01).square()
        self._diagnostics = {"gamma": float(gamma.detach()),
                             "residual_rms_ratio": float(ratio.detach()),
                             **self.cross.last_diagnostics}
        return base + injected


class WWFCAv41UNet(nn.Module):
    """HF UNet with one v4.1 branch parallel to the 32->16 downsample."""
    def __init__(self, enabled: bool, widths: Tuple[int, ...] = (128, 128, 128, 128),
                 layers: int = 2, attention_enabled: bool = True):
        super().__init__(); self.enabled = enabled
        self.unet = UNet2DModel(sample_size=128, in_channels=2, out_channels=1,
            layers_per_block=layers, block_out_channels=widths,
            down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False)
        target = self.unet.down_blocks[-2]
        condition_dim = widths[0] * 4 + 1
        target.downsamplers[0] = WWFCAv41Downsample(
            target.downsamplers[0], widths[-2], condition_dim, enabled,
            attention_enabled=attention_enabled)

    @property
    def adapter(self): return self.unet.down_blocks[-2].downsamplers[0]

    def load_hf_backbone(self, state: dict):
        mapped = {}; prefix = "down_blocks.2.downsamplers.0."
        for key, value in state.items():
            mapped[key.replace(prefix, prefix + "base.", 1) if key.startswith(prefix) else key] = value
        result = self.unet.load_state_dict(mapped, strict=False)
        missing = [key for key in result.missing_keys
                   if ".cross." not in key and not key.endswith("gamma_logit")]
        if result.unexpected_keys or missing:
            raise RuntimeError(f"HF weight mapping failed; missing={missing}, unexpected={result.unexpected_keys}")

    def _time_condition(self, sample, timestep, sigma):
        if not torch.is_tensor(timestep): timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0: timestep = timestep[None]
        timestep = timestep.to(sample.device).expand(sample.shape[0])
        temb = self.unet.time_embedding(self.unet.time_proj(timestep).to(dtype=sample.dtype))
        noise = torch.log1p(sigma.to(device=sample.device, dtype=temb.dtype).clamp_min(0.0))[:, None]
        return torch.cat((temb, noise), dim=1)

    def energy_penalty(self): return self.adapter.energy_penalty()
    def diagnostics(self): return self.adapter.diagnostics()

    def forward(self, sample, timestep, sigma=None):
        sigma = sample.new_zeros(sample.shape[0]) if sigma is None else sigma
        self.adapter.set_condition(self._time_condition(sample, timestep, sigma))
        try: return self.unet(sample, timestep)
        finally: self.adapter.set_condition(None)
