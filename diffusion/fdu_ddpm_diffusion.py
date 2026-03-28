import torch
from diffusers import DDPMScheduler

from model.model_setup import ModelSetup
from selector.diffusion_selector import register_diffusion
import tqdm

@register_diffusion(name='FduDDPMDiffusion')
class FduDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = ModelSetup(self.config, config.logger).model
        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps, prediction_type=self.config.diffusion.prediction_type)

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
        encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
        self.noise = torch.randn_like(self.gt_unwrapped_neg_norm).to(self.device)
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped_neg_norm, self.noise, t).to(self.device)
        model_input = torch.cat([self.noisy] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
        # print("model_input shape:", model_input.shape)
        self.noise_pred = self.model(
            model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        self.pred_unwrapped_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
        if self.config.diffusion.prediction_type == "sample":
            self.pred_batch["pred"] = self.pred_unwrapped_neg_norm
            self.pred_batch["gt"] = self.gt_unwrapped_neg_norm
        elif self.config.diffusion.prediction_type == "v_prediction":
            self.v_pred = self.noise_pred
            self.v_target = self.scheduler.get_velocity(
                self.gt_unwrapped_neg_norm,
                self.noise,
                t
            )
            self.pred_batch["pred"] = self.v_pred
            self.pred_batch["gt"] = self.v_target
        else:
            self.pred_batch["pred"] = self.noise_pred
            self.pred_batch["gt"] = self.noise
        # print(self.noise.max(), self.noise.min())
        # print(self.noise_pred.max(), self.noise_pred.min())

    def infer_sample(self):
        self.scheduler.set_timesteps(self.config.diffusion.num_infer_timesteps)
        with torch.no_grad():
            encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
            # scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps, prediction_type=self.config.diffusion.prediction_type)
            x = torch.randn_like(self.wrapped).to(self.device)
            for t in tqdm.tqdm(self.scheduler.timesteps, desc="Sampling"):
                model_input = torch.cat([x] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
                self.noise_pred = self.model(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                x = self.scheduler.step(self.noise_pred, t, x).prev_sample
            self.pred_unwrapped_neg_norm = x
            self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
            self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
            self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
            self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
            self.pred_batch["pred"] = self.noise_pred
            self.pred_batch["gt"] = self.noise_pred
        self.scheduler.set_timesteps(self.config.diffusion.num_train_timesteps)

    @property
    def optimize_parameters(self):
        return self.model.parameters()