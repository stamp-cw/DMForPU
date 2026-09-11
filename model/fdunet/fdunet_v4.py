from diffusers import UNet2DConditionModel
from .fdunet_2d_blocks import get_mid_block, get_down_block, get_up_block
from selector.model_selector import register_model
import torch.nn as nn

@register_model(name=['FDUNetV4'])
class FDUNetV4(UNet2DConditionModel):
    def __init__(self, config):
        super().__init__(
            sample_size=config.model.sample_size,
            in_channels=config.model.in_channels * config.diffusion.repeat_channels + config.diffusion.conditioning_channels * 1,
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
            cross_attention_dim=config.model.cross_attention_dim,
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )