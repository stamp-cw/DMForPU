import torch
from diffusers import UNet2DModel, DDPMScheduler

from selector.diffusion_selector import register_diffusion
import tqdm


@register_diffusion(name='TextDDPMDiffusion')
class TextDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = UNet2DModel(
            sample_size=128,      # H = W
            in_channels=1,        # 灰度相位图 → 1
            out_channels=1,       # 预测 noise
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps, beta_schedule="linear")

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def setup_data(self, batch_dict):
        self.wrapped = batch_dict["wrapped"].to(self.device)
        # self.wrapped_cond = batch_dict["wrapped_cond"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.gt_unwrapped_norm = batch_dict["unwrapped_neg_norm"].to(self.device)

    def train_sample(self, t):
        self.noise = torch.randn_like(self.gt_unwrapped_norm)
        xt = self.scheduler.add_noise(self.gt_unwrapped_norm, self.noise, t)
        self.noise_pred = self.model(xt, t).sample
        self.pred_unwrapped = self.gt_unwrapped


    def infer_sample(self):
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)
        x = torch.randn_like(self.gt_unwrapped_norm ).to(self.device)

        for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            with torch.no_grad():
                self.noise_pred = self.model(x, t_batch).sample
                x = scheduler.step(self.noise_pred, t, x).prev_sample
        self.pred_unwrapped_neg_norm = x
        self.pred_unwrapped = self.pred_unwrapped_neg_norm

    @property
    def optimize_parameters(self):
        return self.model.parameters()
