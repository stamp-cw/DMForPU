import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from selector.model_selector import register_model

from functools import partial

def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = torch.prod(torch.tensor(shape)) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * torch.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * torch.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.):
    """The same initialization used in DDPM."""
    scale = 1e-10 if scale == 0 else scale
    return variance_scaling(scale, 'fan_avg', 'uniform')


def _einsum(a, b, c, x, y):
    einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
    return torch.einsum(einsum_str, x, y)

def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[:len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)

class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)

class SpatialCrossAttnBlock(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.BatchNorm_0 = nn.BatchNorm2d(channels)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.)

    def forward(self, x_q, x_k, x_v):
        # x_q: bi query
        # x_k: ci key
        # x_v: ci value
        # B, C, H, W = x.shape
        B, C, H, W = x_q.shape
        # h = self.GroupNorm_0(x)
        h_q = self.BatchNorm_0(x_q)
        h_k = self.BatchNorm_0(x_k)
        h_v = self.BatchNorm_0(x_v)
        q = self.NIN_0(h_q)
        k = self.NIN_1(h_k)
        v = self.NIN_2(h_v)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        return x_q + h
        # return x_v + h

import torch.nn as nn
from torch import cat as cat

' Branch0 block '
class Branch0(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch0, self).__init__()
        # self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv0 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.bt0 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bt0(x0)
        # x0 = x0 + self.time_proj(t_emb)[:, :, None, None]
        return x0

' Branch1 block '
class Branch1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Branch1, self).__init__()
        # self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt1 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bt1(x1)
        # x1 = x1 + self.time_proj(t_emb)[:, :, None, None]
        return x1

' Branch2 block '
class Branch2(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(Branch2, self).__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_1 = nn.BatchNorm2d(out_ch)
        self.rl2_1 = nn.LeakyReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt2_2 = nn.BatchNorm2d(out_ch)
    def forward(self, x, t_emb):
        x2 = self.conv2_1(x)
        x2 = self.bt2_1(x2)
        x2 = self.rl2_1(x2)
        x2 = x2 + self.time_proj(t_emb)[:, :, None, None]
        x2 = self.conv2_2(x2)
        x2 = self.bt2_2(x2)
        return x2

' Branch3 block '
class Branch3(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(Branch3, self).__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv3_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_1 = nn.BatchNorm2d(out_ch)
        self.rl3_1 = nn.LeakyReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_2 = nn.BatchNorm2d(out_ch)
        self.rl3_2 = nn.LeakyReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt3_3 = nn.BatchNorm2d(out_ch)
    def forward(self, x, t_emb):
        x3 = self.conv3_1(x)
        x3 = self.bt3_1(x3)
        x3 = self.rl3_1(x3)
        x3 = x3 + self.time_proj(t_emb)[:, :, None, None]
        x3 = self.conv3_2(x3)
        x3 = self.bt3_2(x3)
        x3 = self.rl3_2(x3)
        x3 = x3 + self.time_proj(t_emb)[:, :, None, None]
        x3 = self.conv3_3(x3)
        x3 = self.bt3_3(x3)
        return x3

' Branch4 block '
class Branch4(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(Branch4, self).__init__()
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv4_1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_1 = nn.BatchNorm2d(out_ch)
        self.rl4_1 = nn.LeakyReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_2 = nn.BatchNorm2d(out_ch)
        self.rl4_2 = nn.LeakyReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_3 = nn.BatchNorm2d(out_ch)
        self.rl4_3 = nn.LeakyReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bt4_4 = nn.BatchNorm2d(out_ch)
    def forward(self, x, t_emb):
        x4 = self.conv4_1(x)
        x4 = self.bt4_1(x4)
        x4 = self.rl4_1(x4)
        x4 = x4 + self.time_proj(t_emb)[:, :, None, None]
        x4 = self.conv4_2(x4)
        x4 = self.bt4_2(x4)
        x4 = self.rl4_2(x4)
        x4 = x4 + self.time_proj(t_emb)[:, :, None, None]
        x4 = self.conv4_3(x4)
        x4 = self.bt4_3(x4)
        x4 = self.rl4_3(x4)
        x4 = x4 + self.time_proj(t_emb)[:, :, None, None]
        x4 = self.conv4_4(x4)
        x4 = self.bt4_4(x4)
        return x4

' Residual block with Inception module '
class ResB(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(ResB, self).__init__()
        self.branch0 = Branch0(in_ch, out_ch)
        self.branch1 = Branch1(in_ch, out_ch // 4)
        self.branch2 = Branch2(in_ch, out_ch // 4, time_dim)
        self.branch3 = Branch3(in_ch, out_ch // 4, time_dim)
        self.branch4 = Branch4(in_ch, out_ch // 4, time_dim)
        self.rl = nn.LeakyReLU(inplace=True)
    def forward(self, x, t_emb):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x, t_emb)
        x3 = self.branch3(x, t_emb)
        x4 = self.branch4(x, t_emb)
        x5 = cat((x1, x2, x3, x4), dim=1)
        x6 =  x0 + x5
        x7= self.rl(x6)
        return  x7

# ' Downsampling block '
# class DownB(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DownB, self).__init__()
#         self.res = ResB(in_ch, out_ch)
#         self.pool = nn.MaxPool2d(kernel_size=2)
#     def forward(self, x):
#         x1 = self.res(x)
#         x2 = self.pool(x1)
#         return x2, x1

' Downsampling block '
class DownB(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(DownB, self).__init__()
        self.c_attn = SpatialCrossAttnBlock(in_ch)
        self.res = ResB(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x_q, x_k, x_v, t_emb):
        x0 = self.c_attn(x_q, x_k, x_v)
        x1 = self.res(x0, t_emb)
        x2 = self.pool(x1)
        return x2, x1

# ' Upsampling block '
# class UpB(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(UpB, self).__init__()
#         self.up = nn.ConvTranspose2d(
#             in_ch, out_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1 )
#         self.res = ResB(out_ch*2, out_ch)
#     def forward(self, x, x_):
#         x1 = self.up(x)
#         x2 = cat((x1 , x_), dim=1)
#         x3 = self.res(x2)
#         return x3

' Upsampling block '
class UpB(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super(UpB, self).__init__()
        self.c_attn = SpatialCrossAttnBlock(in_ch)
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=3, stride=2, padding = 1, output_padding = 1 )
        self.res = ResB(out_ch*2, out_ch, time_dim)
    def forward(self, x_q, x_k, x_v, x_, t_emb):
        x0 = self.c_attn(x_q, x_k, x_v)
        x1 = self.up(x0)
        x2 = cat((x1 , x_), dim=1)
        x3 = self.res(x2, t_emb)
        return x3

' Output layer '
class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
    def forward(self, x):
        x1 = self.conv(x)
        return x1


# class TimeEmbedding(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.SiLU(),
#             nn.Linear(dim * 4, dim)
#         )
#
#     def forward(self, t):
#         return self.mlp(t)


def timestep_embedding(timesteps, dim):
    """
    timesteps: (B,)
    return: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000)) * torch.arange(half, device=device) / (half - 1)
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


' Architecture of Res-UNet '
@register_model(name='DiffAuxUNet')
class DiffAuxUNet(nn.Module):
    def __init__(self, config):
        super(DiffAuxUNet, self).__init__()

        time_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        self.down1 = DownB(1, 64, time_dim)
        self.down2 = DownB(64, 128, time_dim)
        self.down3 = DownB(128, 256, time_dim)
        self.down4 = DownB(256, 512, time_dim)
        self.res = ResB(512, 1024, time_dim)
        self.up1 = UpB(1024, 512, time_dim)
        self.up2 = UpB(512, 256, time_dim)
        self.up3 = UpB(256, 128, time_dim)
        self.up4 = UpB(128, 64, time_dim)
        self.outc = Outconv(64, 1)
    def forward(self, x, t, feats=False, down_feats=None, up_feats=None):
        t_emb = timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)

        if feats:
            d_feat_0, d_feat_1, d_feat_2, d_feat_3 = down_feats
            x1, x1_ = self.down1(x, d_feat_0, d_feat_0, t_emb)
            x2, x2_ = self.down2(x1, d_feat_1, d_feat_1, t_emb)
            x3, x3_ = self.down3(x2, d_feat_2, d_feat_2, t_emb)
            x4, x4_ = self.down4(x3, d_feat_3, d_feat_3, t_emb)
            x5  = self.res(x4, t_emb)
            u_feat_0, u_feat_1, u_feat_2, u_feat_3 = up_feats
            x6  = self.up1(x5, u_feat_0, u_feat_0, x4_, t_emb)
            x7  = self.up2(x6, u_feat_1, u_feat_1, x3_, t_emb)
            x8  = self.up3(x7, u_feat_2, u_feat_2, x2_, t_emb)
            x9  = self.up4(x8, u_feat_3, u_feat_3, x1_, t_emb)
        else:
            x1, x1_ = self.down1(x, x, x, t_emb)
            x2, x2_ = self.down2(x1, x1, x1, t_emb)
            x3, x3_ = self.down3(x2, x2, x2, t_emb)
            x4, x4_ = self.down4(x3, x3, x3, t_emb)
            x5  = self.res(x4, t_emb)
            x6  = self.up1(x5, x5, x5, x4_, t_emb)
            x7  = self.up2(x6, x6, x6, x3_, t_emb)
            x8  = self.up3(x7, x7, x7, x2_, t_emb)
            x9  = self.up4(x8, x8, x8, x1_, t_emb)
        x10 = self.outc(x9)
        return x10, (x, x1, x2, x3), (x5, x6, x7, x8)