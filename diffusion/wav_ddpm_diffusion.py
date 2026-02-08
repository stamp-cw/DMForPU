import torch
from diffusers import UNet2DConditionModel, DDPMScheduler

from selector.diffusion_selector import register_diffusion
import tqdm
import torch.nn as nn
from torch.distributions import Normal

class PhaseEncoder(nn.Module):
    def __init__(self, embed_dim=768, patch_size=8):
        super().__init__()
        self.proj = nn.Conv2d(
            1 + 1, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, phase):
        """
        phase: [B, 1, H, W]
        return: [B, N, D]
        """
        x = self.proj(phase)
        x = x.flatten(2).transpose(1, 2)
        return x


@register_diffusion(name='WavDDPMDiffusion')
class WavDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = UNet2DConditionModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels * config.diffusion.repeat_channels + config.diffusion.conditioning_channels * (1 + 3 * self.config.data.wavelet_level),
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
            cross_attention_dim=config.model.cross_attention_dim,
            down_block_types=(
                # "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            )
        ).to(self.device)
        # self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps, prediction_type="sample", clip_sample=False)
        # self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps, prediction_type="sample")
        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)
        self.normal = Normal(0.0, 1.0)

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def setup_data(self, batch_dict):
        self.pred_batch = batch_dict
        self.wrapped = batch_dict["wrapped"].to(self.device)
        self.wrapped_cond = batch_dict["wrapped_cond"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.gt_unwrapped_neg_norm = batch_dict["unwrapped_neg_norm"].to(self.device)

    def train_sample(self, t):
        # cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        # encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim, device=self.device)
        encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
        self.noise = torch.randn_like(self.gt_unwrapped_neg_norm).to(self.device)
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped_neg_norm, self.noise, t).to(self.device)
        model_input = torch.cat([self.noisy] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
        self.noise_pred = self.model(
            model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            # encoder_hidden_states=None,

        ).sample
        self.pred_unwrapped_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        # self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))  - torch.pi
        # self.pred_unwrapped = self.config.data.mean + (self.config.data.std / self.config.data.scale_alpha) * torch.atanh(self.pred_unwrapped_neg_norm)
        # self.pred_unwrapped = self.config.data.mean + (self.config.data.std / self.config.data.scale_alpha) * torch.atanh(self.pred_unwrapped_neg_norm)

        # self.pred_unwrapped = self.config.data.mean + self.config.data.std *  self.normal.icdf(self.pred_unwrapped_norm.clamp(1e-5, 1 - 1e-5))
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
        self.pred_batch["pred"] = self.noise_pred
        self.pred_batch["gt"] = self.noise
        # self.pred_batch["pred"] = self.pred_unwrapped_neg_norm
        # self.pred_batch["gt"] = self.gt_unwrapped_neg_norm
        # print(self.noise.max(), self.noise.min())
        # print(self.noise_pred.max(), self.noise_pred.min())

    def infer_sample(self):
        # cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        # encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim, device=self.device)
        with torch.no_grad():
            encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
            # scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps, prediction_type="sample", clip_sample=False)
            # scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps, prediction_type="sample")
            scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)
            x = torch.randn_like(self.wrapped).to(self.device)
            for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
                model_input = torch.cat([x] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
                self.noise_pred = self.model(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    # encoder_hidden_states=None,
                ).sample
                x = scheduler.step(self.noise_pred, t, x).prev_sample
            self.pred_unwrapped_neg_norm = x
            self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
            self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min)) - torch.pi
            # self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
            # self.pred_unwrapped = self.config.data.mean + (self.config.data.std / self.config.data.scale_alpha) * torch.atanh(self.pred_unwrapped_neg_norm)
            # self.pred_unwrapped = self.config.data.mean + self.config.data.std *  self.normal.icdf(self.pred_unwrapped_norm.clamp(1e-5, 1 - 1e-5))
            self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
            self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
            self.pred_batch["pred"] = self.noise_pred
            self.pred_batch["gt"] = self.noise_pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()
