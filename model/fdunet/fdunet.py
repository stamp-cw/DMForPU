from diffusers import UNet2DConditionModel
from .fdunet_2d_blocks import get_mid_block
from selector.model_selector import register_model


@register_model(name=['FDUNet'])
class FDUNet(UNet2DConditionModel):
    def __init__(self, config):
        super().__init__(
            sample_size=config.model.sample_size,
            # in_channels=config.model.in_channels * config.diffusion.repeat_channels + config.diffusion.conditioning_channels * 1,
            in_channels=config.model.in_channels * config.diffusion.repeat_channels + config.diffusion.conditioning_channels * 1,
            out_channels=config.model.out_channels,
            layers_per_block=config.model.layers_per_block,
            block_out_channels=tuple(config.model.block_out_channels),
            cross_attention_dim=config.model.cross_attention_dim,
            down_block_types=(
                # "CrossAttnDownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        # self.mid_block_type = "FDUNetMidBlock2DCrossAttn"
        # self.blocks_time_embed_dim = config.model.time_embed_dim
        # mid

        # time
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            self.config.time_embedding_type,
            block_out_channels=self.config.block_out_channels,
            flip_sin_to_cos=self.config.flip_sin_to_cos,
            freq_shift=self.config.freq_shift,
            time_embedding_dim=self.config.time_embedding_dim,
        )

        if self.config.class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        if isinstance(self.config.transformer_layers_per_block, int):
            transformer_layers_per_block = [self.config.transformer_layers_per_block] * len(self.config.down_block_types)
        else:
            transformer_layers_per_block = self.config.transformer_layers_per_block


        if isinstance(self.config.attention_head_dim, int):
            attention_head_dim = (self.config.attention_head_dim,)
        else:
            attention_head_dim = self.config.attention_head_dim

        num_attention_heads = self.config.num_attention_heads or attention_head_dim

        if isinstance(self.config.num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(self.config.down_block_types)
        else:
            num_attention_heads = num_attention_heads

        if isinstance(self.config.cross_attention_dim, int):
            cross_attention_dim = (self.config.cross_attention_dim,) * len(self.config.down_block_types)
        else:
            cross_attention_dim = self.config.cross_attention_dim

        if self.config.mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False
        else:
            mid_block_only_cross_attention = self.config.mid_block_only_cross_attention

        if isinstance(self.config.attention_head_dim, int):
            attention_head_dim = (self.config.attention_head_dim,) * len(self.config.down_block_types)
        else:
            attention_head_dim = self.config.attention_head_dim

        self.mid_block = get_mid_block(
            mid_block_type="FDUNetMidBlock2DCrossAttn",
            temb_channels=blocks_time_embed_dim,
            in_channels=self.config.block_out_channels[-1],
            resnet_eps=self.config.norm_eps,
            resnet_act_fn=self.config.act_fn,
            resnet_groups=self.config.norm_num_groups,
            output_scale_factor=self.config.mid_block_scale_factor,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=cross_attention_dim[-1],
            dual_cross_attention=self.config.dual_cross_attention,
            use_linear_projection=self.config.use_linear_projection,
            mid_block_only_cross_attention=mid_block_only_cross_attention,
            upcast_attention=self.config.upcast_attention,
            resnet_time_scale_shift=self.config.resnet_time_scale_shift,
            attention_type=self.config.attention_type,
            resnet_skip_time_act=self.config.resnet_skip_time_act,
            cross_attention_norm=self.config.cross_attention_norm,
            attention_head_dim=attention_head_dim[-1],
            dropout=self.config.dropout,
        )