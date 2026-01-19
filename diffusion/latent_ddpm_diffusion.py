import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, ControlNetModel, AutoencoderKL

from selector.diffusion_selector import register_diffusion
import tqdm


@register_diffusion(name='DDPMDiffusion')
class DDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        if config.diffusion.use_vae:
            model_in_channels = config.vae.in_channels  + (config.diffusion.conditioning_channels if config.diffusion.on_conditioning and self.config.diffusion.fusion_type == 'concat' else 0)
            model_out_channels = config.vae.in_channels
        else:
            model_in_channels = config.model.in_channels + (config.diffusion.conditioning_channels if config.diffusion.on_conditioning and self.config.diffusion.fusion_type == 'concat' else 0)
            model_out_channels = config.model.out_channels
        self.model = UNet2DConditionModel(
            sample_size=config.model.sample_size,
            in_channels=model_in_channels,
            out_channels=model_out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
        ).to(self.device)
        if config.diffusion.use_controlnet:
            self.controlnet_model = ControlNetModel(
                in_channels=config.controlnet_model.in_channels,
                conditioning_channels=config.controlnet_model.conditioning_channels,
                layers_per_block=config.controlnet_model.layers_per_block,
                block_out_channels=tuple(config.controlnet_model.block_out_channels),
            ).to(self.device)

        self.vae = AutoencoderKL(
            in_channels=self.config.vae.in_channels,
            out_channels=self.config.vae.out_channels,
            latent_channels=self.config.vae.latent_channels,
            block_out_channels=tuple(self.config.vae.block_out_channels),
        ).to(self.device)

        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)

    def setup_train(self):
        self.model.train()
        self.vae.train()
        if self.config.diffusion.use_controlnet:
            self.controlnet_model.train()


    def setup_eval(self):
        self.model.eval()
        self.vae.eval()
        if self.config.diffusion.use_controlnet:
            self.controlnet_model.eval()

    def setup_data(self, batch_dict):
        self.wrapped = batch_dict["wrapped"].to(self.device)
        self.wrapped_cond = batch_dict["wrapped_cond"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.gt_unwrapped_norm = batch_dict["unwrapped_neg_norm"].to(self.device)

    def train_sample(self, t):
        if self.config.diffusion.use_vae:
            with torch.no_grad():
                self.latents = self.vae.encode(self.gt_unwrapped_norm).latent_dist.sample()
                self.latents = self.latents * self.config.diffusion.scaling_factor
        else:
            self.latents = self.gt_unwrapped_norm
        cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim,
                                                                           device=self.device)
        self.noise = torch.randn_like(self.latents).to(self.device)
        self.noisy = self.scheduler.add_noise(self.latents, self.noise, t).to(self.device)
        if self.config.diffusion.on_conditioning:
            if self.config.diffusion.use_controlnet:
                control_cond = self.wrapped_cond
                B, C, H_latent, W_latent = self.noisy.shape
                downsample_factor = 2 ** (len(self.controlnet_model.config.block_out_channels) - 1)
                control_cond = torch.nn.functional.interpolate(control_cond,
                                                               size=(H_latent * downsample_factor, W_latent * downsample_factor),
                                                               mode="bilinear", align_corners=False)
                ctrl_down, ctrl_mid = self.controlnet_model(
                    self.noisy,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_cond.to(self.device),
                    return_dict=False,
                )
                self.noise_pred = self.model(
                    self.noisy,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals= ctrl_down,
                    mid_block_additional_residual=ctrl_mid,
                ).sample
            else:
                # print(f"noisy shape: {self.noisy.shape}, wrapped_cond shape: {self.wrapped_cond.shape}")
                # model_input = torch.cat([self.noisy, self.wrapped_cond], dim=1)
                if self.config.diffusion.fusion_type == 'concat':
                    model_input = torch.cat([self.noisy, self.wrapped_cond], dim=1)
                else:
                    model_input = torch.clamp((self.noisy + self.wrapped_cond[0] + self.wrapped_cond[1]) / 3, -1, 1)
                # print(model_input.shape)
                self.noise_pred = self.model(
                    model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
        else:
            self.noise_pred = self.model(
                self.noisy,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
        self.pred_latents =  self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        if self.config.diffusion.use_vae:
            self.pred_latents = self.pred_latents / self.config.diffusion.scaling_factor
            self.pred_unwrapped_neg_norm = self.vae.decode(self.pred_latents).sample
        else:
            self.pred_unwrapped_neg_norm = self.pred_latents
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.scale_k)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    def infer_sample(self):
        if self.config.diffusion.use_vae:
            with torch.no_grad():
                self.latents = self.vae.encode(self.gt_unwrapped_norm).latent_dist.sample()
                self.latents = self.latents * self.config.diffusion.scaling_factor
        else:
            self.latents = self.gt_unwrapped_norm
        cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim, device=self.device)
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)
        if self.config.diffusion.on_conditioning:
            if self.config.diffusion.use_controlnet:
                control_cond = self.wrapped_cond
                x = torch.randn_like(self.wrapped).to(self.device)
                B, C, H_latent, W_latent = x.shape
                downsample_factor = 2 ** (len(self.controlnet_model.config.block_out_channels) - 1)
                control_cond = torch.nn.functional.interpolate(control_cond,
                                                               size=(H_latent * downsample_factor, W_latent * downsample_factor),
                                                               mode="bilinear", align_corners=False)
                for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
                    ctrl_down, ctrl_mid = self.controlnet_model(
                        x,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=control_cond.to(self.device),
                        return_dict=False,
                    )
                    self.noise_pred = self.model(
                        x,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=ctrl_down,
                        mid_block_additional_residual=ctrl_mid,
                    ).sample
                    x = scheduler.step(self.noise_pred, t, x).prev_sample
            else:
                x = torch.randn_like(self.wrapped).to(self.device)
                for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
                    if self.config.diffusion.fusion_type == 'concat':
                        model_input = torch.cat([x, self.wrapped_cond], dim=1)
                    else:
                        model_input = torch.clamp((x + self.wrapped_cond[0] + self.wrapped_cond[1]) / 3, -1, 1)
                    self.noise_pred = self.model(
                        model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    x = scheduler.step(self.noise_pred, t, x).prev_sample

        else:
            x = torch.randn_like(self.wrapped).to(self.device)
            for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
                self.noise_pred = self.model(
                    x,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
                x = scheduler.step(self.noise_pred, t, x).prev_sample
        self.pred_latents =  x
        if self.config.diffusion.use_vae:
            self.pred_latents = self.pred_latents / self.config.diffusion.scaling_factor
            self.pred_unwrapped_neg_norm = self.vae.decode(self.pred_latents).sample
        else:
            self.pred_unwrapped_neg_norm = self.pred_latents
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.scale_k)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    @property
    def optimize_parameters(self):
        if self.config.diffusion.use_controlnet:
            return list(self.model.parameters())+ list(self.controlnet_model.parameters())
        else:
            return self.model.parameters()
