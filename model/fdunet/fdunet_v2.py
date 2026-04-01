from diffusers import UNet2DConditionModel
from .fdunet_2d_blocks import get_mid_block, get_down_block, get_up_block
from selector.model_selector import register_model
import torch.nn as nn

@register_model(name=['FDUNetV2'])
class FDUNetV2(UNet2DConditionModel):
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
                "DownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "UpBlock2D",
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

        num_attention_heads = self.config.num_attention_heads or attention_head_dim[0]

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(self.config.down_block_types)

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
        # down
        if isinstance(self.config.layers_per_block, int):
            layers_per_block = [self.config.layers_per_block] * len(self.config.down_block_types)

        if isinstance(self.config.only_cross_attention, bool):
            only_cross_attention = [self.config.only_cross_attention] * len(self.config.down_block_types)

        output_channel = self.config.block_out_channels[0]
        down_block_types = list(self.config.down_block_types)
        # print(f"Down block types before modification: {down_block_types}")
        down_block_types[-1] = "FDCrossAttnDownBlock2D"
        down_block_types=tuple(down_block_types)
        self.down_blocks = nn.ModuleList([])
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = self.config.block_out_channels[i]
            is_final_block = i == len(self.config.block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                resnet_groups=self.config.norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=self.config.downsample_padding,
                dual_cross_attention=self.config.dual_cross_attention,
                use_linear_projection=self.config.use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=self.config.upcast_attention,
                resnet_time_scale_shift=self.config.resnet_time_scale_shift,
                attention_type=self.config.attention_type,
                resnet_skip_time_act=self.config.resnet_skip_time_act,
                resnet_out_scale_factor=self.config.resnet_out_scale_factor,
                cross_attention_norm=self.config.cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=self.config.dropout,
            )
            self.down_blocks.append(down_block)

        # up
        reversed_block_out_channels = list(reversed(self.config.block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if self.config.reverse_transformer_layers_per_block is None
            else self.config.reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        up_block_types=list(self.config.up_block_types)
        up_block_types[0] = "FDCrossAttnUpBlock2D"
        up_block_types = tuple(up_block_types)
        self.up_blocks = nn.ModuleList([])
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(self.config.block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(self.config.block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=self.config.norm_eps,
                resnet_act_fn=self.config.act_fn,
                resolution_idx=i,
                resnet_groups=self.config.norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=self.config.dual_cross_attention,
                use_linear_projection=self.config.use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=self.config.upcast_attention,
                resnet_time_scale_shift=self.config.resnet_time_scale_shift,
                attention_type=self.config.attention_type,
                resnet_skip_time_act=self.config.resnet_skip_time_act,
                resnet_out_scale_factor=self.config.resnet_out_scale_factor,
                cross_attention_norm=self.config.cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=self.config.dropout,
            )
            self.up_blocks.append(up_block)