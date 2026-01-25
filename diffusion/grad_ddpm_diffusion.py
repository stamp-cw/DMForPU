import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, ControlNetModel

from selector.diffusion_selector import register_diffusion
import tqdm

from utils.util import poisson_reconstruct_phase


@register_diffusion(name='GradDDPMDiffusion')
class GradDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = UNet2DConditionModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels * config.diffusion.repeat_channels + config.diffusion.conditioning_channels,
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
        ).to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)

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
        self.gt_unwrapped_grad_neg_norm = batch_dict["unwrapped_grad_neg_norm"].to(self.device)


    def train_sample(self, t):
        cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim, device=self.device)
        self.noise = torch.randn_like(self.gt_unwrapped_grad_neg_norm).to(self.device)
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped_grad_neg_norm, self.noise, t).to(self.device)
        # print(f"noisy shape: {self.noisy.shape}, wrapped_cond shape: {self.wrapped_cond.shape}")
        model_input = torch.cat([self.noisy] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
        self.noise_pred = self.model(
            model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        self.pred_unwrapped_grad_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        # self.pred_unwrapped_grad = self.pred_unwrapped_grad_neg_norm * 2
        self.pred_unwrapped_grad = self.pred_unwrapped_grad_neg_norm / 10
        pred_unwrapped_grad_x, pred_unwrapped_grad_y = torch.chunk(self.pred_unwrapped_grad, chunks=2, dim=1)
        self.pred_unwrapped_neg_norm = poisson_reconstruct_phase(pred_unwrapped_grad_x, pred_unwrapped_grad_y)
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
        self.pred_batch["pred_unwrapped_grad_neg_norm"] = self.pred_unwrapped_grad_neg_norm
        self.pred_batch["pred"] = self.noise_pred
        self.pred_batch["gt"] = self.noise

    def infer_sample(self):
        cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        encoder_hidden_states = None if cross_dim is None else torch.zeros(self.gt_unwrapped_grad_neg_norm.shape[0], 1, cross_dim, device=self.device)
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)
        x = torch.randn_like(self.gt_unwrapped_grad_neg_norm).to(self.device)
        for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
            model_input = torch.cat([x] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
            self.noise_pred = self.model(
                model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                # encoder_hidden_states=None,
            ).sample
            x = scheduler.step(self.noise_pred, t, x).prev_sample
        # self.pred_unwrapped_neg_norm = x
        self.pred_unwrapped_grad_neg_norm = x
        # self.pred_unwrapped_grad = self.pred_unwrapped_grad_neg_norm * 2
        self.pred_unwrapped_grad = self.pred_unwrapped_grad_neg_norm / 10
        pred_unwrapped_grad_x, pred_unwrapped_grad_y = torch.chunk(self.pred_unwrapped_grad, chunks=2, dim=1)
        self.pred_unwrapped_neg_norm = poisson_reconstruct_phase(pred_unwrapped_grad_x, pred_unwrapped_grad_y)
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
        self.pred_batch["pred_unwrapped_grad_neg_norm"] = self.pred_unwrapped_grad_neg_norm
        self.pred_batch["pred"] = self.noise_pred
        self.pred_batch["gt"] = self.noise_pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()
