"""WWFCA-v4.2: v4.1 frequency branch with monotonic timestep/noise residual gating."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import UNet2DModel

from .fdunet_atten import _haar_dwt2
from .wwfca_v41 import GlobalFrequencyGatedAttention, _logit


class WWFCAv42Downsample(nn.Module):
    """Original HF downsample plus a sample-wise gamma(t, sigma) frequency residual."""

    def __init__(self, base: nn.Module, channels: int, condition_dim: int,
                 enabled: bool, use_noise_gate: bool, max_scale: float = 0.1):
        super().__init__(); self.base = base; self.enabled = enabled
        self.use_noise_gate = use_noise_gate; self.max_scale = max_scale
        self.cross = GlobalFrequencyGatedAttention(channels, condition_dim)
        self.gamma_bias = nn.Parameter(torch.tensor(_logit(0.2)))  # approximately 0.02
        # Bounded non-negative strengths. Initial 0.05 makes v4.2 start close
        # to the constant-gamma v4.1 while retaining useful gradients.
        self.time_logit = nn.Parameter(torch.tensor(_logit(0.05 / 4.0)))
        self.noise_logit = nn.Parameter(torch.tensor(_logit(0.05 / 4.0)))
        self._condition = None; self._progress = None; self._noise = None
        self._energy_penalty = None
        self._diagnostics = {"gamma": 0.0, "gamma_min": 0.0, "gamma_max": 0.0,
            "time_strength": 0.0, "noise_strength": 0.0, "residual_rms_ratio": 0.0,
            **self.cross.last_diagnostics}

    def set_conditions(self, condition, progress, noise):
        self._condition = condition; self._progress = progress; self._noise = noise

    def energy_penalty(self):
        return self.gamma_bias.new_zeros(()) if self._energy_penalty is None else self._energy_penalty

    def diagnostics(self): return dict(self._diagnostics)

    def forward(self, x):
        base = self.base(x)
        if not self.enabled:
            self._energy_penalty = base.new_zeros(())
            self._diagnostics = {key: 0.0 for key in self._diagnostics}
            return base
        ll, lh, hl, hh = _haar_dwt2(x)
        residual = self.cross(ll, lh, hl, hh, self._condition)
        time_strength = 4.0 * torch.sigmoid(self.time_logit)
        noise_strength = 4.0 * torch.sigmoid(self.noise_logit) if self.use_noise_gate else self.noise_logit.new_zeros(())
        logits = self.gamma_bias + time_strength * self._progress
        if self.use_noise_gate: logits = logits - noise_strength * self._noise
        gamma = self.max_scale * torch.sigmoid(logits)
        injected = gamma[:, None, None, None] * residual
        ratio = (injected.float().square().mean().sqrt() /
                 base.float().square().mean().sqrt().clamp_min(1e-8))
        # Diagnostic only. The trainer deliberately excludes this from v4.2 loss.
        self._energy_penalty = F.relu(ratio - 0.01).square()
        self._diagnostics = {"gamma": float(gamma.detach().mean()),
            "gamma_min": float(gamma.detach().min()), "gamma_max": float(gamma.detach().max()),
            "time_strength": float(time_strength.detach()),
            "noise_strength": float(noise_strength.detach()),
            "residual_rms_ratio": float(ratio.detach()), **self.cross.last_diagnostics}
        return base + injected


class WWFCAv42UNet(nn.Module):
    def __init__(self, enabled: bool, use_noise_gate: bool,
                 widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__(); self.enabled = enabled; self.use_noise_gate = use_noise_gate
        self.unet = UNet2DModel(sample_size=128, in_channels=2, out_channels=1,
            layers_per_block=layers, block_out_channels=widths,
            down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False)
        target = self.unet.down_blocks[-2]; condition_dim = widths[0] * 4 + 1
        target.downsamplers[0] = WWFCAv42Downsample(
            target.downsamplers[0], widths[-2], condition_dim, enabled, use_noise_gate)

    @property
    def adapter(self): return self.unet.down_blocks[-2].downsamplers[0]

    def load_hf_backbone(self, state: dict):
        mapped = {}; prefix = "down_blocks.2.downsamplers.0."
        for key, value in state.items():
            mapped[key.replace(prefix, prefix + "base.", 1) if key.startswith(prefix) else key] = value
        result = self.unet.load_state_dict(mapped, strict=False)
        adapter_names = (".cross.", "gamma_bias", "time_logit", "noise_logit")
        missing = [key for key in result.missing_keys if not any(token in key for token in adapter_names)]
        if result.unexpected_keys or missing:
            raise RuntimeError(f"HF weight mapping failed; missing={missing}, unexpected={result.unexpected_keys}")

    def _conditions(self, sample, timestep, sigma):
        if not torch.is_tensor(timestep): timestep = torch.tensor([timestep], device=sample.device)
        if timestep.ndim == 0: timestep = timestep[None]
        timestep = timestep.to(sample.device).expand(sample.shape[0])
        temb = self.unet.time_embedding(self.unet.time_proj(timestep).to(dtype=sample.dtype))
        noise = torch.log1p(sigma.to(device=sample.device, dtype=temb.dtype).clamp_min(0.0))
        condition = torch.cat((temb, noise[:, None]), dim=1)
        progress = 1.0 - timestep.to(dtype=temb.dtype) / 999.0
        return condition, progress.clamp(0.0, 1.0), noise

    def energy_penalty(self): return self.adapter.energy_penalty()
    def diagnostics(self): return self.adapter.diagnostics()

    def forward(self, sample, timestep, sigma=None):
        sigma = sample.new_zeros(sample.shape[0]) if sigma is None else sigma
        condition, progress, noise = self._conditions(sample, timestep, sigma)
        self.adapter.set_conditions(condition, progress, noise)
        try: return self.unet(sample, timestep)
        finally: self.adapter.set_conditions(None, None, None)
