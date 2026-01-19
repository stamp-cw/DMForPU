import torch
from diffusers import AutoencoderKL

from selector.vae_selector import register_vae


@register_vae(name='LatentVAE')
class LatentVAE:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.model = AutoencoderKL(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            latent_channels=self.config.model.latent_channels,
            block_out_channels=tuple(self.config.model.block_out_channels),
        ).to(self.device)

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def train_predict(self, gt):
        self.gt = gt.to(self.device)
        self.posterior = self.model.encode(self.gt).latent_dist
        z = self.posterior.sample()
        self.pred = self.model.decode(z).sample
        return self.pred

    def eval_predict(self, gt):
        with torch.no_grad():
            self.gt = gt.to(self.device)
            self.gt = self.gt.to(self.device)
            self.posterior = self.model.encode(self.gt).latent_dist
            z = self.posterior.mode()
            print("z mean:", z.mean().item(), "std:", z.std().item())
            self.pred = self.model.decode(z).sample
        return self.pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()