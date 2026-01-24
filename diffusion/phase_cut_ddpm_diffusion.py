import torch
from diffusers import DDPMScheduler, UNet2DConditionModel

from selector.diffusion_selector import register_diffusion
import tqdm
import torch.nn as nn

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


@register_diffusion(name='PhaseCutDDPMDiffusion')
class PhaseCutDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = UNet2DConditionModel(
            sample_size=32,
            in_channels=1,     # φ_wrap
            out_channels=1,    # φ_unwrap
            cross_attention_dim=768,
            block_out_channels=(64, 128, 256, 256),
        ).to(self.device)

        self.phase_encoder = PhaseEncoder(embed_dim=768, patch_size=8).to(self.device)

        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)

    def setup_train(self):
        self.model.train()
        self.phase_encoder.train()

    def setup_eval(self):
        self.model.eval()
        self.phase_encoder.eval()

    def setup_data(self, batch_dict):
        self.pred_batch = batch_dict
        self.wrapped = batch_dict["wrapped"].to(self.device)
        self.wrapped_cond = batch_dict["wrapped_cond"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.gt_unwrapped_std_norm = batch_dict["unwrapped_std_norm"].to(self.device)

    def train_sample(self, t):

        self.phase_emb = self.phase_encoder(self.wrapped_cond)
        self.noise = torch.randn_like(self.gt_unwrapped_std_norm)
        x_t = self.scheduler.add_noise(self.gt_unwrapped_std_norm, self.noise, t)

        self.noise_pred = self.model(
            x_t,
            t,
            encoder_hidden_states=self.phase_emb
        ).sample
        self.pred_unwrapped_std_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), x_t).pred_original_sample
        # self.pred_unwrapped = self.pred_unwrapped_std_norm * self.config.data.std + self.config.data.mean
        self.pred_unwrapped = ((self.pred_unwrapped_std_norm + 1 ) /2 ) * (3 * 2  * torch.pi)
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_std_norm"] = self.pred_unwrapped_std_norm
        self.pred_batch["pred"] = self.noise_pred
        self.pred_batch["gt"] = self.noise


    def infer_sample(self):
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)
        x = torch.randn_like(self.gt_unwrapped_std_norm ).to(self.device)
        self.phase_emb = self.phase_encoder(self.wrapped_cond)
        for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            with torch.no_grad():
                self.noise_pred = self.noise_pred = self.model(
                    x,
                    t_batch,
                    encoder_hidden_states=self.phase_emb
                ).sample
                x = scheduler.step(self.noise_pred, t, x).prev_sample
        self.pred_unwrapped_std_norm = x
        # self.pred_unwrapped = self.pred_unwrapped_std_norm * self.config.data.std + self.config.data.mean
        self.pred_unwrapped = ((self.pred_unwrapped_std_norm + 1 ) /2 ) * (3 * 2  * torch.pi)
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_std_norm"] = self.pred_unwrapped_std_norm
        self.pred_batch["pred"] = self.noise_pred
        self.pred_batch["gt"] = self.noise

    @property
    def optimize_parameters(self):
        return list(self.model.parameters()) + list(self.phase_encoder.parameters())

