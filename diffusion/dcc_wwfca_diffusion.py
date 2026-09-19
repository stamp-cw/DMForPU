"""Matched HF/DCC/WWFCA diffusion models for controlled ablation studies."""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import torch
from torch import nn
from diffusers import DDPMScheduler, UNet2DModel

from diffusion.chen_hf_diffusion import ChenHFConfig


class DCCWWFCADiffusion(nn.Module):
    VARIANTS = ("hf_matched", "dcc_only", "wwfca_only", "fdu", "dwhfa", "dwhfa_v2", "wwfca_v2_off", "wwfca_v2",
                "wwfca_v3_off", "wwfca_v3", "wwfca_v4_off", "wwfca_v4",
                "wwfca_v41_off", "wwfca_v41_noattn", "wwfca_v41", "wwfca_v42_time", "wwfca_v42",
                "wwfca_v43")

    def __init__(self, variant: str, widths=(128, 128, 128, 128), layers=2, cross_dim=384,
                 phase_low=-14 * math.pi, phase_high=14 * math.pi):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown DCC/WWFCA variant: {variant}")
        self.variant = variant
        self.cfg = ChenHFConfig(physics=False, phase_low=phase_low, phase_high=phase_high)
        self.spec = {"variant": variant, "widths": list(widths), "layers": layers, "cross_dim": cross_dim,
                     "phase_low": phase_low, "phase_high": phase_high}
        self.dcc = variant in ("dcc_only", "fdu")
        if variant == "dwhfa":
            from model.fdunet.dwhfa import DWHFAUNet
            self.backbone = DWHFAUNet(widths=tuple(widths), layers=layers)
        elif variant == "dwhfa_v2":
            from model.fdunet.dwhfa import DWHFAv2UNet
            self.backbone = DWHFAv2UNet(widths=tuple(widths), layers=layers)
        elif variant == "wwfca_v43":
            from model.fdunet.wwfca_v43 import WWFCAv43UNet
            self.backbone = WWFCAv43UNet(widths=tuple(widths), layers=layers)
        elif variant in ("wwfca_v42_time", "wwfca_v42"):
            from model.fdunet.wwfca_v42 import WWFCAv42UNet
            self.backbone = WWFCAv42UNet(
                enabled=True, use_noise_gate=variant == "wwfca_v42",
                widths=tuple(widths), layers=layers
            )
        elif variant in ("wwfca_v41_off", "wwfca_v41_noattn", "wwfca_v41"):
            from model.fdunet.wwfca_v41 import WWFCAv41UNet
            self.backbone = WWFCAv41UNet(
                enabled=variant != "wwfca_v41_off",
                attention_enabled=variant != "wwfca_v41_noattn",
                widths=tuple(widths), layers=layers
            )
        elif variant in ("wwfca_v4_off", "wwfca_v4"):
            from model.fdunet.wwfca_v4 import WWFCAv4UNet
            self.backbone = WWFCAv4UNet(
                enabled=variant == "wwfca_v4", widths=tuple(widths), layers=layers
            )
        elif variant in ("wwfca_v3_off", "wwfca_v3"):
            from model.fdunet.wwfca_v3 import WWFCAv3UNet
            self.backbone = WWFCAv3UNet(
                enabled=variant == "wwfca_v3", widths=tuple(widths), layers=layers
            )
        elif variant in ("wwfca_v2_off", "wwfca_v2"):
            from model.fdunet.wwfca_v2 import WWFCAv2UNet
            self.backbone = WWFCAv2UNet(
                enabled=variant == "wwfca_v2", widths=tuple(widths), layers=layers
            )
        elif variant in ("fdu", "wwfca_only"):
            from model.fdunet.fdunet import FDUNet
            config = NS(
                model=NS(sample_size=128, in_channels=1, out_channels=1, layers_per_block=layers,
                         block_out_channels=list(widths), cross_attention_dim=cross_dim),
                diffusion=NS(repeat_channels=1, conditioning_channels=2 if self.dcc else 1),
            )
            self.backbone = FDUNet(config)
        else:
            self.backbone = UNet2DModel(
                sample_size=128, in_channels=3 if self.dcc else 2, out_channels=1,
                layers_per_block=layers, block_out_channels=tuple(widths),
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                add_attention=False,
            )
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)

    def normalize(self, phase):
        return 2 * (phase - self.cfg.phase_low) / (self.cfg.phase_high - self.cfg.phase_low) - 1

    def denormalize(self, value):
        return self.cfg.phase_low + (value + 1) * (self.cfg.phase_high - self.cfg.phase_low) / 2

    def config_dict(self):
        result = {"architecture": self.spec, "prediction_type": "sample", "clip_sample": False,
                  "train_steps": 1000, "inference_steps": 5, "dcc": self.dcc,
                  "wwfca": self.variant in ("wwfca_only", "fdu", "dwhfa", "dwhfa_v2", "wwfca_v2", "wwfca_v3", "wwfca_v4", "wwfca_v41_noattn", "wwfca_v41", "wwfca_v42_time", "wwfca_v42", "wwfca_v43"),
                  "cross_attention": self.variant in ("wwfca_only", "fdu", "wwfca_v2", "wwfca_v3", "wwfca_v4", "wwfca_v41", "wwfca_v42_time", "wwfca_v42", "wwfca_v43")}
        if self.variant in ("wwfca_v2_off", "wwfca_v2"):
            result.update(wwfca_version=2, wwfca_enabled=self.variant == "wwfca_v2")
        elif self.variant in ("wwfca_v3_off", "wwfca_v3"):
            result.update(wwfca_version=3, wwfca_enabled=self.variant == "wwfca_v3",
                          max_residual_scale=0.1, sigma_encoding="log1p",
                          insertion="mid-only", frequency_direction="H<-LL", reconstruction="Haar IDWT")
        elif self.variant in ("wwfca_v4_off", "wwfca_v4"):
            result.update(wwfca_version=4, wwfca_enabled=self.variant == "wwfca_v4",
                          max_residual_scale=0.1, initial_residual_scale=0.02,
                          windowed=False, insertion="parallel penultimate downsample (32->16)",
                          query="LL", key_value="Conv(Concat(LH,HL,HH))", reconstruction="direct residual")
        elif self.variant in ("wwfca_v41_off", "wwfca_v41_noattn", "wwfca_v41"):
            result.update(wwfca_version="4.1", wwfca_enabled=self.variant != "wwfca_v41_off",
                          max_residual_scale=0.1, initial_residual_scale=0.02,
                          windowed=False, insertion="parallel penultimate downsample (32->16)",
                          query="LL", key_value="separate Conv(LH/HL/HH)",
                          attention="disabled; fixed G=0.5" if self.variant == "wwfca_v41_noattn" else "global cosine with temperature and distance prior",
                          fusion="attention-gated spatial high-frequency residual")
        elif self.variant == "dwhfa":
            result.update(wwfca_version="DWHFA", wwfca_enabled=True,
                          insertion="parallel penultimate downsample (32->16)",
                          decomposition="Haar DWT; LL unused",
                          directional_encoder="LH 3x1, HL 1x3, HH 3x3 depthwise+pointwise conv",
                          attention="channel-spatial sigmoid prior; no Q/K/V attention",
                          calibration="Y=X+alpha*X*(2A-1)", alpha_init=0.1,
                          identity_init="zero-initialized attention logits")
        elif self.variant == "dwhfa_v2":
            result.update(wwfca_version="DWHFA-v2", wwfca_enabled=True,
                          insertion="first three UNet downsamplers (128→64, 64→32, 32→16)",
                          decomposition="Haar DWT at each inserted stage; LL unused",
                          directional_encoder="LH 3x1, HL 1x3, HH 3x3 depthwise+pointwise conv",
                          attention="channel-spatial sigmoid prior; no Q/K/V attention",
                          calibration="Y=X+alpha*X*(2A-1) at each stage", alpha_init=0.1,
                          identity_init="zero-initialized attention logits at each stage")
        elif self.variant in ("wwfca_v42_time", "wwfca_v42"):
            result.update(wwfca_version="4.2", wwfca_enabled=True,
                          max_residual_scale=0.1, windowed=False,
                          insertion="parallel penultimate downsample (32->16)",
                          query="LL", key_value="separate Conv(LH/HL/HH)",
                          attention="global cosine with temperature and distance prior",
                          fusion="attention-gated spatial high-frequency residual",
                          residual_gate="monotonic gamma(t,sigma)" if self.variant == "wwfca_v42" else "monotonic gamma(t)")
        elif self.variant == "wwfca_v43":
            result.update(wwfca_version="4.3", wwfca_enabled=True,
                          insertion="parallel penultimate downsample (32->16)",
                          validation="H queries X; confidence-gated directional high frequency",
                          retrieval="X queries validated H",
                          fusion="local high-frequency skip plus retrieved residual",
                          residual_gate="constant learned gamma")
        return result

    def forward(self, noisy, wrapped, timestep, sigma):
        condition = torch.cat((wrapped.sin(), wrapped.cos()), 1) if self.dcc else wrapped / math.pi
        model_input = torch.cat((noisy, condition), 1)
        if self.variant in ("wwfca_v2_off", "wwfca_v2", "wwfca_v3_off", "wwfca_v3",
                            "wwfca_v4_off", "wwfca_v4", "wwfca_v41_off", "wwfca_v41_noattn", "wwfca_v41",
                            "wwfca_v42_time", "wwfca_v42", "wwfca_v43", "dwhfa", "dwhfa_v2"):
            estimate = self.backbone(model_input, timestep, sigma=sigma).sample.float()
        elif self.variant in ("wwfca_only", "fdu"):
            hidden = torch.zeros(len(wrapped), 1, self.spec["cross_dim"], device=wrapped.device, dtype=wrapped.dtype)
            estimate = self.backbone(model_input, timestep, encoder_hidden_states=hidden).sample.float()
        else:
            estimate = self.backbone(model_input, timestep).sample.float()
        phase = self.denormalize(estimate)
        return estimate, phase, [phase]

    @torch.no_grad()
    def sample(self, wrapped, sigma, generator=None, steps=None):
        scheduler = DDPMScheduler.from_config(self.scheduler.config)
        scheduler.set_timesteps(steps or 5, device=wrapped.device)
        value = torch.randn(wrapped.shape, device=wrapped.device, dtype=wrapped.dtype, generator=generator)
        for timestep in scheduler.timesteps:
            prediction, _, _ = self(value, wrapped, timestep, sigma)
            value = scheduler.step(prediction, timestep, value, generator=generator).prev_sample
        return self.denormalize(value)
