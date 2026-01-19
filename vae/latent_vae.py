import torch
from diffusers import AutoencoderKL

from selector.diffusion_selector import register_diffusion


@register_diffusion(name='LatentVAE')
class LatentVAE:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = AutoencoderKL(
            in_channels=self.config.vae.in_channels,
            out_channels=self.config.vae.out_channels,
            latent_channels=self.config.vae.latent_channels,
            block_out_channels=tuple(self.config.vae.block_out_channels),
        )

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def predict(self, gt):
        self.gt = gt.to(self.device)
        self.posterior = self.model.encode(self.gt)
        z = self.posterior.sample()
        self.pred = self.model.decode(z).sample
        return self.pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()