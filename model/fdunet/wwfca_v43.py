"""WWFCA-v4.3: validate high frequencies, then retrieve them bidirectionally."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch import nn

from .fdunet_atten import _haar_dwt2


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


class SpatialCrossAttention(nn.Module):
    """Cosine cross-attention with an explicit local-distance prior."""

    def __init__(self, channels: int, heads: int = 4, max_size: int = 16,
                 dropout: float = 0.05):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.max_size = max_size
        self.dropout = dropout
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Linear(channels, channels, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, 2 * max_size - 1, 2 * max_size - 1))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)
        self.temperature_logit = nn.Parameter(torch.tensor(_logit(3.0 / 15.0)))
        self.distance_logit = nn.Parameter(torch.tensor(_logit(1.0 / 4.0)))

    def _geometry(self, height: int, width: int, device, dtype):
        if height > self.max_size or width > self.max_size:
            raise ValueError(f"attention grid {height}x{width} exceeds {self.max_size}")
        yy, xx = torch.meshgrid(torch.arange(height, device=device),
                                torch.arange(width, device=device), indexing="ij")
        coords = torch.stack((yy.flatten(), xx.flatten()))
        relative = coords[:, :, None] - coords[:, None, :]
        iy = relative[0] + self.max_size - 1
        ix = relative[1] + self.max_size - 1
        bias = self.relative_bias[:, iy, ix].to(dtype=dtype)
        dy = relative[0].float() / max(height - 1, 1)
        dx = relative[1].float() / max(width - 1, 1)
        distance = torch.sqrt(dy.square() + dx.square()).to(dtype=dtype)
        return bias, distance

    def forward(self, query_map, context_map):
        batch, channels, height, width = query_map.shape
        if context_map.shape != query_map.shape:
            raise ValueError(f"query/context mismatch: {query_map.shape} vs {context_map.shape}")
        query = self.query_norm(query_map.flatten(2).transpose(1, 2))
        context = self.context_norm(context_map.flatten(2).transpose(1, 2))
        q = self.to_q(query).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q = F.normalize(q.float(), dim=-1).to(query.dtype)
        k = F.normalize(k.float(), dim=-1).to(context.dtype)
        bias, distance = self._geometry(height, width, q.device, q.dtype)
        temperature = 1 + 15 * torch.sigmoid(self.temperature_logit)
        distance_strength = 4 * torch.sigmoid(self.distance_logit)
        logits = temperature * torch.matmul(q, k.transpose(-2, -1))
        logits = logits + bias[None] - distance_strength * distance[None, None]
        attention = logits.float().softmax(dim=-1).to(logits.dtype)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch, height * width, channels)
        output = self.to_out(output).transpose(1, 2).reshape(batch, channels, height, width)
        entropy = -(attention.float().clamp_min(1e-12) *
                    attention.float().clamp_min(1e-12).log()).sum(-1)
        diagnostics = {
            "entropy": float((entropy / math.log(height * width)).mean().detach()),
            "temperature": float(temperature.detach()),
            "distance_strength": float(distance_strength.detach()),
        }
        return output, diagnostics


class ValidateThenRetrieve(nn.Module):
    """H->X validates directional H; X->H retrieves the validated content."""

    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.band_convs = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(min(32, channels), channels), nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv2d(channels, channels, 1),
            ) for _ in range(3)
        ])
        self.high_fuse = nn.Conv2d(3 * channels, channels, 1)
        self.condition_norm = nn.LayerNorm(condition_dim)
        self.film = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.validate = SpatialCrossAttention(channels)
        self.validation_gate = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1), nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        # A very small nonzero initialization keeps the gate near 0.5 while
        # allowing the H->X validation attention to receive gradients from
        # the first optimization step under mixed precision.
        nn.init.xavier_uniform_(self.validation_gate[-1].weight, gain=0.01)
        nn.init.zeros_(self.validation_gate[-1].bias)
        self.validation_scale_logit = nn.Parameter(torch.tensor(_logit(0.2)))
        self.retrieve = SpatialCrossAttention(channels)
        self.retrieval_scale_logit = nn.Parameter(torch.tensor(_logit(0.2)))
        self.output_norm = nn.GroupNorm(min(32, channels), channels)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.xavier_uniform_(self.output.weight, gain=0.1)
        nn.init.zeros_(self.output.bias)
        self.last_diagnostics = {
            "validation_entropy": 0.0, "retrieval_entropy": 0.0,
            "validation_gate_mean": 0.5, "validation_gate_spatial_std": 0.0,
            "validation_scale": 0.05, "retrieval_scale": 0.1,
        }

    def forward(self, x, lh, hl, hh, condition: Optional[torch.Tensor] = None):
        directional = [conv(band) for conv, band in zip(self.band_convs, (lh, hl, hh))]
        high = self.high_fuse(torch.cat(directional, dim=1))
        if condition is not None:
            scale, shift = self.film(self.condition_norm(condition)).chunk(2, dim=1)
            high = high * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        # Validation: each high-frequency token asks whether X supports it.
        support, validation_diag = self.validate(high, x)
        gate = torch.sigmoid(self.validation_gate(torch.cat((high, support, (high - support).abs()), dim=1)))
        validation_scale = 0.25 * torch.sigmoid(self.validation_scale_logit)
        validated_high = high * (0.5 + gate) + validation_scale * support

        # Retrieval: X asks the validated high-frequency bank for useful detail.
        retrieved, retrieval_diag = self.retrieve(x, validated_high)
        retrieval_scale = 0.5 * torch.sigmoid(self.retrieval_scale_logit)
        residual = validated_high + retrieval_scale * retrieved
        output = self.output(self.output_norm(residual))
        with torch.no_grad():
            self.last_diagnostics = {
                "validation_entropy": validation_diag["entropy"],
                "retrieval_entropy": retrieval_diag["entropy"],
                "validation_gate_mean": float(gate.float().mean()),
                "validation_gate_spatial_std": float(gate.float().std(dim=(-2, -1)).mean()),
                "validation_scale": float(validation_scale.detach()),
                "retrieval_scale": float(retrieval_scale.detach()),
            }
        return output


class WWFCAv43Downsample(nn.Module):
    def __init__(self, base: nn.Module, channels: int, condition_dim: int,
                 max_scale: float = 0.1):
        super().__init__()
        self.base = base
        self.cross = ValidateThenRetrieve(channels, condition_dim)
        self.max_scale = max_scale
        self.gamma_logit = nn.Parameter(torch.tensor(_logit(0.2)))
        self._condition = None
        self._energy_penalty = None
        self._diagnostics = {"gamma": 0.0, "residual_rms_ratio": 0.0,
                             **self.cross.last_diagnostics}

    def set_condition(self, condition):
        self._condition = condition

    def energy_penalty(self):
        return self.gamma_logit.new_zeros(()) if self._energy_penalty is None else self._energy_penalty

    def diagnostics(self):
        return dict(self._diagnostics)

    def forward(self, x):
        base = self.base(x)
        _, lh, hl, hh = _haar_dwt2(x)
        residual = self.cross(base, lh, hl, hh, self._condition)
        gamma = self.max_scale * torch.sigmoid(self.gamma_logit)
        injected = gamma * residual
        ratio = (injected.float().square().mean().sqrt() /
                 base.float().square().mean().sqrt().clamp_min(1e-8))
        self._energy_penalty = F.relu(ratio - 0.01).square()
        self._diagnostics = {"gamma": float(gamma.detach()),
                             "residual_rms_ratio": float(ratio.detach()),
                             **self.cross.last_diagnostics}
        return base + injected


class WWFCAv43UNet(nn.Module):
    def __init__(self, widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=128, in_channels=2, out_channels=1,
            layers_per_block=layers, block_out_channels=widths,
            down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False,
        )
        target = self.unet.down_blocks[-2]
        condition_dim = widths[0] * 4 + 1
        target.downsamplers[0] = WWFCAv43Downsample(
            target.downsamplers[0], widths[-2], condition_dim)

    @property
    def adapter(self):
        return self.unet.down_blocks[-2].downsamplers[0]

    def load_hf_backbone(self, state: dict):
        mapped = {}
        prefix = "down_blocks.2.downsamplers.0."
        for key, value in state.items():
            mapped[key.replace(prefix, prefix + "base.", 1) if key.startswith(prefix) else key] = value
        result = self.unet.load_state_dict(mapped, strict=False)
        missing = [key for key in result.missing_keys
                   if ".cross." not in key and not key.endswith("gamma_logit")]
        if result.unexpected_keys or missing:
            raise RuntimeError(f"HF weight mapping failed; missing={missing}, unexpected={result.unexpected_keys}")

    def _time_condition(self, sample, timestep, sigma):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(sample.device).expand(sample.shape[0])
        temb = self.unet.time_embedding(self.unet.time_proj(timestep).to(dtype=sample.dtype))
        noise = torch.log1p(sigma.to(device=sample.device, dtype=temb.dtype).clamp_min(0.0))[:, None]
        return torch.cat((temb, noise), dim=1)

    def energy_penalty(self):
        return self.adapter.energy_penalty()

    def diagnostics(self):
        return self.adapter.diagnostics()

    def forward(self, sample, timestep, sigma=None):
        sigma = sample.new_zeros(sample.shape[0]) if sigma is None else sigma
        self.adapter.set_condition(self._time_condition(sample, timestep, sigma))
        try:
            return self.unet(sample, timestep)
        finally:
            self.adapter.set_condition(None)
