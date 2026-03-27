import torch
from diffusers import DDPMScheduler, ControlNetModel

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
        self.controlnet_model = ControlNetModel(
            in_channels=3,
            conditioning_channels=2,
            layers_per_block=2,
            # block_out_channels=(64, 64, 64, 64),
            block_out_channels=(32, 32, 32, 32),
            # cross_attention_dim=192,
            cross_attention_dim=96,
            down_block_types=(
                # "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            )
        ).to(self.device)

    def setup_train(self):
        self.model.train()
        self.controlnet_model.train()

    def setup_eval(self):
        self.model.eval()
        self.controlnet_model.eval()

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
        control_cond = self.wrapped_cond
        B, C, H_latent, W_latent = control_cond.shape
        downsample_factor = 2 ** (len(self.controlnet_model.config.block_out_channels) - 1)
        control_cond = torch.nn.functional.interpolate(control_cond, size=(H_latent * downsample_factor, W_latent * downsample_factor), mode="bilinear", align_corners=False)
        # print(f"model_input: {model_input.shape}")
        # print(f"control_cond: {control_cond.shape}")
        ctrl_down, ctrl_mid = self.controlnet_model(
            model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_cond.to(self.device),
            return_dict=False,
        )
        self.noise_pred = self.model(
            model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals = ctrl_down,
            mid_block_additional_residual = ctrl_mid
        ).sample
        # self.v_pred = self.noise_pred
        # alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        # alpha_bar = alphas_cumprod[t[0].cpu()].view(-1, 1, 1, 1)
        #
        # self.v_target = (
        #         torch.sqrt(alpha_bar) * self.noise
        #         - torch.sqrt(1 - alpha_bar) * self.gt_unwrapped_neg_norm
        # )
        self.pred_unwrapped_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * (self.config.data.k_max - self.config.data.k_min))
        # self.pred_unwrapped = self.config.data.mean + self.config.data.std *  self.normal.icdf(self.pred_unwrapped_norm.clamp(1e-5, 1 - 1e-5))
        self.pred_batch["pred_unwrapped"] = self.pred_unwrapped
        self.pred_batch["pred_unwrapped_neg_norm"] = self.pred_unwrapped_neg_norm
        if self.config.diffusion.prediction_type == "sample":
            self.pred_batch["pred"] = self.pred_unwrapped_neg_norm
            self.pred_batch["gt"] = self.gt_unwrapped_neg_norm
        # elif self.config.diffusion.prediction_type == "v_prediction":
        #     self.pred_batch["pred"] = self.v_pred
        #     self.pred_batch["gt"] = self.v_target
        else:
            self.pred_batch["pred"] = self.noise_pred
            self.pred_batch["gt"] = self.noise
        # print(self.noise.max(), self.noise.min())
        # print(self.noise_pred.max(), self.noise_pred.min())

    def infer_sample(self):
        self.scheduler.set_timesteps(self.config.diffusion.num_infer_timesteps)

        with torch.no_grad():
            # encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
            # if self.config.diffusion.use_patch_embedding:
            #     encoder_hidden_states = self.phase_encoder(self.wrapped_cond)
            # else:
            #     encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
            encoder_hidden_states = torch.zeros(self.wrapped.shape[0], 1, self.config.model.cross_attention_dim, device=self.device)
            x = torch.randn_like(self.wrapped).to(self.device)
            control_cond = self.wrapped_cond
            B, C, H_latent, W_latent = control_cond.shape
            downsample_factor = 2 ** (len(self.controlnet_model.config.block_out_channels) - 1)
            control_cond = torch.nn.functional.interpolate(control_cond, size=(H_latent * downsample_factor,
                                                                               W_latent * downsample_factor),
                                                           mode="bilinear", align_corners=False)
            for t in tqdm.tqdm(self.scheduler.timesteps, desc="Sampling"):
                model_input = torch.cat([x] * self.config.diffusion.repeat_channels + [self.wrapped_cond], dim=1)
                ctrl_down, ctrl_mid = self.controlnet_model(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_cond.to(self.device),
                    return_dict=False,
                )
                self.noise_pred = self.model(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=ctrl_down,
                    mid_block_additional_residual=ctrl_mid,
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
        if self.config.diffusion.use_controlnet:
            return list(self.model.parameters()) + list(self.controlnet_model.parameters())
        else:
            return self.model.parameters()