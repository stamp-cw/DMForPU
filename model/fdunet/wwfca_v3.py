"""Bounded and progressively enabled WWFCA v3 components."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from diffusers import UNet2DModel

from .wwfca_v2 import WWFCAHighFromLow


class MidBlockWithWWFCAV3(nn.Module):
    """A mid-only frequency residual with normalized conditions and LayerScale."""

    def __init__(
        self,
        base: nn.Module,
        channels: int,
        temb_channels: int,
        enabled: bool,
        max_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.base = base
        self.enabled = enabled
        self.max_residual_scale = float(max_residual_scale)
        self.frequency = WWFCAHighFromLow(channels, dropout=0.05)
        self.condition_norm = nn.LayerNorm(temb_channels + 1)
        self.gate = nn.Sequential(
            nn.Linear(temb_channels + 1, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.register_buffer("gate_scale", torch.tensor(1.0))
        self._sigma: Optional[torch.Tensor] = None
        self._gate_regularization: Optional[torch.Tensor] = None
        self._diagnostics = {
            "gate_mean_abs": 0.0,
            "gate_max_abs": 0.0,
            "gate_saturated_fraction": 0.0,
            "frequency_residual_rms_ratio": 0.0,
        }

    @staticmethod
    def encode_sigma(sigma: torch.Tensor) -> torch.Tensor:
        """Continuous encoding with clean sigma=0 mapped exactly to zero."""
        return torch.log1p(sigma.clamp_min(0.0))

    def set_condition(self, sigma: Optional[torch.Tensor], gate_scale: Optional[float] = None) -> None:
        self._sigma = sigma
        if gate_scale is not None:
            self.gate_scale.fill_(float(gate_scale))

    def gate_regularization(self) -> torch.Tensor:
        if self._gate_regularization is None:
            return self.gate_scale.new_zeros(())
        return self._gate_regularization

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs):
        hidden_states = self.base(hidden_states, temb, *args, **kwargs)
        if not self.enabled or float(self.gate_scale) == 0.0:
            self._gate_regularization = hidden_states.new_zeros(())
            self._diagnostics = {key: 0.0 for key in self._diagnostics}
            return hidden_states
        if temb is None:
            raise ValueError("WWFCA v3 requires the diffusion timestep embedding")
        sigma = temb.new_zeros(len(temb)) if self._sigma is None else self._sigma.to(temb)
        noise_condition = self.encode_sigma(sigma)[:, None]
        condition = self.condition_norm(torch.cat((temb, noise_condition), dim=1))
        raw_gate = torch.tanh(self.gate(condition))
        alpha = self.max_residual_scale * self.gate_scale.to(raw_gate) * raw_gate
        frequency = self.frequency(hidden_states)
        injected = alpha[:, :, None, None] * frequency
        self._gate_regularization = raw_gate.square().mean()
        with torch.no_grad():
            denominator = hidden_states.float().square().mean().sqrt().clamp_min(1e-8)
            self._diagnostics = {
                "gate_mean_abs": float(alpha.detach().abs().mean()),
                "gate_max_abs": float(alpha.detach().abs().max()),
                "gate_saturated_fraction": float((raw_gate.detach().abs() > 0.8).float().mean()),
                "frequency_residual_rms_ratio": float(injected.detach().float().square().mean().sqrt() / denominator),
            }
        return hidden_states + injected


class WWFCAv3UNet(nn.Module):
    """Exact HF backbone plus an optional, parameter-matched WWFCA-v3 branch."""

    def __init__(self, enabled: bool, widths: Tuple[int, ...] = (128, 128, 128, 128), layers: int = 2):
        super().__init__()
        self.enabled = enabled
        self.unet = UNet2DModel(
            sample_size=128, in_channels=2, out_channels=1, layers_per_block=layers,
            block_out_channels=widths, down_block_types=("DownBlock2D",) * len(widths),
            up_block_types=("UpBlock2D",) * len(widths), add_attention=False,
        )
        self.unet.mid_block = MidBlockWithWWFCAV3(
            self.unet.mid_block, widths[-1], widths[0] * 4, enabled=enabled
        )

    def set_gate_scale(self, value: float) -> None:
        self.unet.mid_block.gate_scale.fill_(float(value))

    def gate_regularization(self) -> torch.Tensor:
        return self.unet.mid_block.gate_regularization()

    def gate_diagnostics(self) -> dict:
        return self.unet.mid_block.diagnostics()

    def forward(self, sample: torch.Tensor, timestep, sigma: Optional[torch.Tensor] = None):
        mid = self.unet.mid_block
        mid.set_condition(sigma)
        try:
            return self.unet(sample, timestep)
        finally:
            mid.set_condition(None)
