import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, ControlNetModel

from model.sec_model_setup import SecModelSetup
from selector.diffusion_selector import register_diffusion
import tqdm

from utils.util import ExtraTokenCondition


@register_diffusion(name='DphDDPMDiffusion')
class DphDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        # self.unet = UNet2DConditionModel(
        #     sample_size=config.model.sample_size,
        #     in_channels=config.model.in_channels + (config.diffusion.conditioning_channels if config.diffusion.on_conditioning and self.config.diffusion.fusion_type == 'concat' else 0),
        #     out_channels=config.model.out_channels,
        #     layers_per_block=config.model.layers_per_block,
        #     block_out_channels=tuple(config.model.block_out_channels),
        # ).to(self.device)

        self.unet = UNet2DConditionModel(
            sample_size=128,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=256,
        ).to(self.device)

        self.model = self.unet

        self.aux_unet = SecModelSetup(self.config, config.logger).model

        # load aux_unet weights to aux_unet
        # self.aux_unet.load_state_dict(torch.load(self.config.model.aux_unet_path, map_location=self.device))
        # self.aux_unet.eval()

        self.token_condition = ExtraTokenCondition(
            self.config,
            # in_channels_list=[256, 256],  # down3 + mid
            in_channels_list=[256, 512],  # down3 + mid
            cross_attention_dim=self.unet.config.cross_attention_dim
        ).to(self.device)

        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)



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
        # cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        # encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim,
        #                                                                    device=self.device)

        aux_feats = self.aux_unet(self.wrapped)
        selected_feats = {
            # "down3": aux_feats["down3"],
            "down2": aux_feats["down2"],
            "mid": aux_feats["mid"],
        }

        encoder_hidden_states = self.token_condition(
            encoder_hidden_states=None,
            feats_dict=selected_feats
        )

        self.noise = torch.randn_like(self.gt_unwrapped_norm).to(self.device)
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped_norm, self.noise, t).to(self.device)
        self.noise_pred = self.model(
            self.noisy,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        self.pred_unwrapped_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.k_max - self.config.data.k_min)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    def infer_sample(self):
        # cross_dim = getattr(self.model.config, "cross_attention_dim", None)
        # encoder_hidden_states = None if cross_dim is None else torch.zeros(self.wrapped.shape[0], 1, cross_dim, device=self.device)
        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)


        # aux_feats = self.aux_unet(self.wrapped)
        aux_feats = self.aux_unet(self.gt_unwrapped_norm)
        selected_feats = {
            # "down3": aux_feats["down3"],
            "down2": aux_feats["down2"],
            "mid": aux_feats["mid"],
        }

        encoder_hidden_states = self.token_condition(
            encoder_hidden_states=None,
            feats_dict=selected_feats
        )


        x = torch.randn_like(self.wrapped).to(self.device)
        for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
            self.noise_pred = self.model(
                x,
                t,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            x = scheduler.step(self.noise_pred, t, x).prev_sample
        self.pred_unwrapped_neg_norm = x
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.k_max - self.config.data.k_min)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    @property
    def optimize_parameters(self):
        if self.config.diffusion.use_controlnet:
            return list(self.model.parameters())+ list(self.controlnet_model.parameters())
        else:
            return self.model.parameters()
