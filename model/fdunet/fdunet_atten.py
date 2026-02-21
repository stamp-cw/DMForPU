from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn as nn
from diffusers import Transformer2DModel
from diffusers.models import DualTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention_processor import Attention
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2DCrossAttn
import torch
from diffusers.utils import logging
import torch.nn.functional as F
logger = logging.get_logger(__name__)

class FDUNetMidBlock2DCrossAttn(UNetMidBlock2DCrossAttn):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            num_layers: int = 1,
            transformer_layers_per_block: Union[int, Tuple[int]] = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_groups_out: Optional[int] = None,
            resnet_pre_norm: bool = True,
            num_attention_heads: int = 1,
            output_scale_factor: float = 1.0,
            cross_attention_dim: int = 1280,
            dual_cross_attention: bool = False,
            use_linear_projection: bool = False,
            upcast_attention: bool = False,
            attention_type: str = "default",
    ):
        super().__init__(
            in_channels=in_channels,
            temb_channels=temb_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_groups_out=resnet_groups_out,
            resnet_pre_norm=resnet_pre_norm,
            num_attention_heads=num_attention_heads,
            output_scale_factor=output_scale_factor,
            cross_attention_dim=cross_attention_dim,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )

        out_channels = out_channels or in_channels

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * num_layers

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        resnet_groups_out = resnet_groups_out or resnet_groups

        attentions = []

        for i in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    FDTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups_out,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        )
                )
        self.attentions = nn.ModuleList(attentions)

class FDTransformer2DModel(Transformer2DModel):
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            interpolation_scale: float = None,
            use_additional_conditions: Optional[bool] = None,
    ):
        super().__init__(
            num_attention_heads = num_attention_heads,
            attention_head_dim = attention_head_dim,
            in_channels = in_channels,
            out_channels = out_channels,
            num_layers = num_layers,
            dropout = dropout,
            norm_num_groups = norm_num_groups,
            cross_attention_dim = cross_attention_dim,
            attention_bias = attention_bias,
            sample_size = sample_size,
            num_vector_embeds = num_vector_embeds,
            patch_size = patch_size,
            activation_fn = activation_fn,
            num_embeds_ada_norm = num_embeds_ada_norm,
            use_linear_projection = use_linear_projection,
            only_cross_attention = only_cross_attention,
            double_self_attention = double_self_attention,
            upcast_attention = upcast_attention,
            norm_type = norm_type,
            norm_elementwise_affine = norm_elementwise_affine,
            norm_eps = norm_eps,
            attention_type = attention_type,
            caption_channels = caption_channels,
            interpolation_scale = interpolation_scale,
            use_additional_conditions = use_additional_conditions,
        )

    def _init_continuous_input(self, norm_type):
        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                FDBasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

class FDBasicTransformerBlock(BasicTransformerBlock):
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = "default",
            positional_embeddings: Optional[str] = None,
            num_positional_embeddings: Optional[int] = None,
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_type=norm_type,
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            attention_type=attention_type,
            positional_embeddings=positional_embeddings,
            num_positional_embeddings=num_positional_embeddings,
            ada_norm_continous_conditioning_embedding_dim=ada_norm_continous_conditioning_embedding_dim,
            ada_norm_bias=ada_norm_bias,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # print("="*100)
        # print(f"x_shape: {hidden_states.shape}")
        # print(f"encoder_hidden_states_shape: {encoder_hidden_states.shape}")
        # print("="*100)

        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)


            # print("="*100)
            # print(f"x_shape: {norm_hidden_states.shape}")
            # print(f"encoder_hidden_states_shape: {encoder_hidden_states.shape}")
            # print("="*100)

            x = norm_hidden_states

            B, N, D = x.shape  # (32,256,64)
            H, W = int(sqrt(N)), int(sqrt(N))

            # 1. BND → BCHW
            x = x.transpose(1,2).contiguous()
            x = x.view(B, D, H, W) # (32,64,8,8)

            # 2. window partition
            window_size, window_stride = int(sqrt(N))//2, int(sqrt(N))//2
            windows = x.unfold(2, window_size, window_stride).unfold(3, window_size, window_stride) # (32,64,2,2,8,8)
            windows = windows.contiguous().view(B*4, D, H//2, W//2) # (32*4,64,8,8)

            # 3. Haar DWT
            LL = (windows[:,:,0::2,0::2] + windows[:,:,1::2,0::2] +
                  windows[:,:,0::2,1::2] + windows[:,:,1::2,1::2]) / 4

            LH = (windows[:,:,0::2,0::2] - windows[:,:,1::2,0::2] +
                  windows[:,:,0::2,1::2] - windows[:,:,1::2,1::2]) / 4

            HL = (windows[:,:,0::2,0::2] + windows[:,:,1::2,0::2] -
                  windows[:,:,0::2,1::2] - windows[:,:,1::2,1::2]) / 4

            HH = (windows[:,:,0::2,0::2] - windows[:,:,1::2,0::2] -
                  windows[:,:,0::2,1::2] + windows[:,:,1::2,1::2]) / 4

            LL_up = F.interpolate(LL, scale_factor=2, mode='nearest')
            LH_up = F.interpolate(LH, scale_factor=2, mode='nearest')
            HL_up = F.interpolate(HL, scale_factor=2, mode='nearest')
            HH_up = F.interpolate(HH, scale_factor=2, mode='nearest')

            # x_q = LL
            x_q = LL_up
            # x_k = torch.cat([LH, HL, HH], dim=1)  # (B*4,64*3,8,8)
            x_k = torch.cat([LH_up , HL_up, HH_up], dim=1)  # (B*4,64*3,8,8)

            # 4. 合并窗口
            x_q = x_q.view(B, 4, D, H//2, W//2) # (B,4,64,8,8)
            x_q = x_q.permute(0, 1, 3, 4, 2).contiguous() # (B,4,8,8,64)
            x_q = x_q.view(B, N, D) # (B,256,64)
            x_k = x_k.view(B, 4, D*3, H//2, W//2) # (B,4,64*3,8,8)
            x_k = x_k.permute(0, 1, 3, 4, 2).contiguous() # (B,4,8,8,64*3)
            x_k = x_k.view(B, N, D*3) # (B,256,64*3)

            attn_output = self.attn2(
                # norm_hidden_states,
                x_q,
                # encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states=x_k,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

