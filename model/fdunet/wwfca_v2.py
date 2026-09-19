"""Conservative, architecture-matched WWFCA v2 components."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from diffusers import UNet2DModel

from .fdunet_atten import _haar_dwt2, _merge_windows, _partition_windows


def haar_idwt2(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Inverse of the orthonormal Haar transform used by WWFCA."""
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("All inverse-Haar bands must have identical shapes")
    batch, channels, height, width = ll.shape
    output = ll.new_empty(batch, channels, height * 2, width * 2)
    output[:, :, 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
    output[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    output[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    output[:, :, 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
    return output


class WWFCAHighFromLow(nn.Module):
    """Use reliable LL context to correct the directional high bands."""

    def __init__(self, channels: int, heads: int = 8, window_size: int = 8, dropout: float = 0.05):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.window_size = window_size
        self.high_norm = nn.LayerNorm(channels)
        self.low_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.band_embedding = nn.Parameter(torch.empty(3, channels))
        nn.init.normal_(self.band_embedding, std=0.02)
        self.fuse = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.SiLU(), nn.Dropout(dropout), nn.Linear(channels, channels)
        )
        self.output = nn.Conv2d(channels, channels, 1)

    def _window(self, height: int, width: int) -> int:
        upper = min(self.window_size, height, width)
        for size in range(upper, 1, -1):
            if size % 2 == 0 and height % size == 0 and width % size == 0:
                return size
        raise ValueError(f"No even WWFCA-v2 window tiles {height}x{width}")

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        window = self._window(height, width)
        windows, num_h, num_w = _partition_windows(feature, window)
        ll, lh, hl, hh = _haar_dwt2(windows)
        reduced = window // 2
        spatial_tokens = reduced * reduced
        low = self.low_norm(ll.flatten(2).transpose(1, 2))
        high_bands = torch.stack((lh, hl, hh), dim=1)
        high = high_bands.permute(0, 1, 3, 4, 2).reshape(-1, 3 * spatial_tokens, channels)
        band_ids = self.band_embedding[:, None, :].expand(3, spatial_tokens, channels)
        query = self.high_norm(high + band_ids.reshape(1, 3 * spatial_tokens, channels))
        context, _ = self.attention(query, low, low, need_weights=False)
        delta = self.fuse(torch.cat((high, context), dim=-1))
        delta = delta.reshape(-1, 3, reduced, reduced, channels).permute(0, 1, 4, 2, 3)
        zero = torch.zeros_like(ll)
        correction = haar_idwt2(zero, delta[:, 0], delta[:, 1], delta[:, 2])
        return _merge_windows(self.output(correction), batch, num_h, num_w)


class MidBlockWithWWFCAV2(nn.Module):
    """Wrap an unchanged HF mid block with a condition-aware gated residual."""

    def __init__(self, base: nn.Module, channels: int, temb_channels: int, enabled: bool):
        super().__init__()
        self.base = base
        self.enabled = enabled
        self.frequency = WWFCAHighFromLow(channels)
        self.gate = nn.Sequential(
            nn.Linear(temb_channels + 1, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self._sigma: Optional[torch.Tensor] = None

    def set_sigma(self, sigma: Optional[torch.Tensor]) -> None:
        self._sigma = sigma

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs):
        hidden_states = self.base(hidden_states, temb, *args, **kwargs)
        if not self.enabled:
            return hidden_states
        if temb is None:
            raise ValueError("WWFCA v2 requires the diffusion timestep embedding")
        if self._sigma is None:
            log_sigma = temb.new_zeros((len(temb), 1))
        else:
            log_sigma = self._sigma.to(device=temb.device, dtype=temb.dtype).clamp_min(1e-6).log()[:, None]
        alpha = torch.tanh(self.gate(torch.cat((temb, log_sigma), dim=1)))[:, :, None, None]
        return hidden_states + alpha * self.frequency(hidden_states)


class WWFCAv2UNet(nn.Module):
    """The exact HF backbone with an optional, parameter-matched mid-only branch."""

    def __init__(self, enabled: bool, widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__()
        self.enabled = enabled
        self.unet = UNet2DModel(
            sample_size=128, in_channels=2, out_channels=1, layers_per_block=layers,
            block_out_channels=widths, down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False,
        )
        self.unet.mid_block = MidBlockWithWWFCAV2(
            self.unet.mid_block, widths[-1], widths[0] * 4, enabled=enabled
        )

    def forward(self, sample: torch.Tensor, timestep, sigma: Optional[torch.Tensor] = None):
        mid = self.unet.mid_block
        mid.set_sigma(sigma)
        try:
            return self.unet(sample, timestep)
        finally:
            mid.set_sigma(None)
