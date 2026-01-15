import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, ControlNetModel

from selector.diffusion_selector import register_diffusion


@register_diffusion(name='DDPMDiffusion')
class DDPMDiffusion:
    def __init__(self, config):
        # self.name = config.diffusion.name
        self.device = config.training.device
        self.model = UNet2DConditionModel(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
        ).to(self.device)
        self.control_model = ControlNetModel(
            in_channels=config.controlnet_model.in_channels,
            conditioning_channels=config.controlnet_model.conditioning_channels,
            layers_per_block=config.controlnet_model.layers_per_block,
            block_out_channels=tuple(config.controlnet_model.block_out_channels),
        ).to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)

        # self.scheduler.set_timesteps(config.diffusion.num_train_timesteps)

    def setup_train(self):
        self.model.train()
        self.control_model.train()

    def setup_eval(self):
        self.model.eval()
        self.control_model.eval()

    def setup_data(self, batch_dict):
        self.wrapped = batch_dict["wrapped"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)


    def sample(self):
        self.noise = torch.randn_like(self.gt_unwrapped).to(self.device)
        t_batch = torch.randint(0, self.scheduler.config.num_train_timesteps, (1,), device=self.device).long()
        t = t_batch.expand(self.wrapped.shape[0])
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped, self.noise, t).to(self.device)
        cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim,
                                                                           device=self.device)
        control_cond = self.wrapped
        B, C, H_latent, W_latent = self.noisy.shape
        downsample_factor = 2 ** (len(self.control_model.config.block_out_channels) - 1)
        # (H_latent * downsample_factor, W_latent * downsample_factor)
        control_cond = torch.nn.functional.interpolate(control_cond, size=(H_latent * downsample_factor,
                                                                           W_latent * downsample_factor),
                                                       mode="bilinear", align_corners=False)
        ctrl_down, ctrl_mid = self.control_model(
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
            down_block_additional_residuals=ctrl_down,
            mid_block_additional_residual=ctrl_mid,
        ).sample
        self.pred_unwrapped = self.scheduler.step(self.noise_pred, t_batch[0].cpu(), self.noisy).prev_sample

    @property
    def optimize_parameters(self):
        return list(self.model.parameters())+ list(self.control_model.parameters())
