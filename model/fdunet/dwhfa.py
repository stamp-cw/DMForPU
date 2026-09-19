"""Directional Wavelet High-Frequency Attention (DWHFA).

The directional high-frequency bands produce a channel-spatial calibration
prior for the original feature X.  They are never injected as content:
Y = X + alpha * X * (2 * sigmoid(A_logits) - 1).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from diffusers import UNet2DModel

from .fdunet_atten import _haar_dwt2


class DWHFADownsample(nn.Module):
    def __init__(self, base: nn.Module, channels: int, alpha_init: float = 0.1):
        super().__init__()
        groups = min(32, channels)
        # With the project's Haar convention LH is vertical high-frequency and
        # HL is horizontal high-frequency, hence 3x1 and 1x3 respectively.
        self.direction_lh = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(groups, channels), nn.SiLU())
        self.direction_hl = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(groups, channels), nn.SiLU())
        self.direction_hh = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1), nn.GroupNorm(groups, channels), nn.SiLU())
        self.fuse = nn.Conv2d(3 * channels, channels, 1)
        self.fuse_norm = nn.GroupNorm(groups, channels)
        self.fuse_act = nn.SiLU()
        self.attention = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.base = base
        self._diagnostics = {"attention_mean": 0.5, "attention_std": 0.0,
                             "centered_mean": 0.0, "centered_std": 0.0,
                             "alpha": float(alpha_init), "residual_ratio": 0.0,
                             "positive_ratio": 0.0, "negative_ratio": 0.0}

    def diagnostics(self):
        return dict(self._diagnostics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, lh, hl, hh = _haar_dwt2(x)
        high = self.fuse(torch.cat((self.direction_lh(lh), self.direction_hl(hl), self.direction_hh(hh)), dim=1))
        high = self.fuse_act(self.fuse_norm(high))
        logits = F.interpolate(self.attention(high), size=x.shape[-2:], mode="bilinear", align_corners=False)
        attention = torch.sigmoid(logits)
        centered = 2.0 * attention - 1.0
        calibrated = x + self.alpha.to(dtype=x.dtype) * x * centered
        residual = calibrated - x
        with torch.no_grad():
            self._diagnostics = {
                "attention_mean": float(attention.detach().mean()),
                "attention_std": float(attention.detach().std()),
                "centered_mean": float(centered.detach().mean()),
                "centered_std": float(centered.detach().std()),
                "alpha": float(self.alpha.detach()),
                "residual_ratio": float(residual.float().square().mean().sqrt() / x.float().square().mean().sqrt().clamp_min(1e-8)),
                "positive_ratio": float((centered > 0).float().mean()),
                "negative_ratio": float((centered < 0).float().mean()),
            }
        return self.base(calibrated)


class DWHFAUNet(nn.Module):
    """HF UNet with DWHFA calibration before the penultimate downsample."""
    def __init__(self, widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__()
        self.unet = UNet2DModel(sample_size=128, in_channels=2, out_channels=1,
            layers_per_block=layers, block_out_channels=widths,
            down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False)
        target = self.unet.down_blocks[-2]
        target.downsamplers[0] = DWHFADownsample(target.downsamplers[0], widths[-2])

    @property
    def adapter(self):
        return self.unet.down_blocks[-2].downsamplers[0]

    def diagnostics(self):
        return self.adapter.diagnostics()

    def forward(self, sample, timestep, sigma=None):
        return self.unet(sample, timestep)


class DWHFAv2UNet(nn.Module):
    """HF UNet with DWHFA calibration at the first three downsampling stages.

    The native UNet remains unchanged apart from replacing the downsampler in
    ``down_blocks[0]``, ``down_blocks[1]`` and ``down_blocks[2]``.  Each stage
    keeps its own directional wavelet encoder and learnable calibration scale.
    """
    def __init__(self, widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__()
        self.unet = UNet2DModel(sample_size=128, in_channels=2, out_channels=1,
            layers_per_block=layers, block_out_channels=widths,
            down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False)
        # DWHFA is applied at 128->64, 64->32 and 32->16 transitions.
        for index in (0, 1, 2):
            target = self.unet.down_blocks[index]
            target.downsamplers[0] = DWHFADownsample(
                target.downsamplers[0], widths[index])

    @property
    def adapters(self):
        return tuple(self.unet.down_blocks[index].downsamplers[0]
                     for index in (0, 1, 2))

    def diagnostics(self):
        return {f"stage_{index}": adapter.diagnostics()
                for index, adapter in zip((0, 1, 2), self.adapters)}

    def forward(self, sample, timestep, sigma=None):
        return self.unet(sample, timestep)
