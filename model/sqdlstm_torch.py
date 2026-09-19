"""Native PyTorch implementation of DeepPhaseUnwrap's JointConvSQDLSTMNet."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        # Keras BatchNormalization defaults: epsilon=1e-3, momentum=0.99.
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )


class JointConvSQDLSTMNetTorch(nn.Module):
    """Layer-for-layer NCHW port of the official channels-last Keras model."""

    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(1, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        self.pool = nn.AvgPool2d(2)

        self.horizontal = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.vertical = nn.LSTM(128, 32, batch_first=True, bidirectional=True)
        self.h_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.v_conv = nn.Conv2d(64, 64, 3, padding=1)

        self.up5 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.c5 = ConvBlock(256, 128)
        self.up6 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.c6 = ConvBlock(128, 64)
        self.up7 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.c7 = ConvBlock(64, 32)
        self.up8 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.c8 = ConvBlock(32, 32)
        self.out = nn.Conv2d(32, 1, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        conv_blocks = (self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7, self.c8)
        for block in conv_blocks:
            nn.init.kaiming_normal_(block[0].weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(block[0].bias)
            nn.init.ones_(block[1].weight); nn.init.zeros_(block[1].bias)
        for conv in (self.h_conv, self.v_conv):
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        # Keras defaults to Glorot for transposed/output convolutions.
        for conv in (self.up5, self.up6, self.up7, self.up8, self.out):
            nn.init.xavier_uniform_(conv.weight); nn.init.zeros_(conv.bias)
        for lstm in (self.horizontal, self.vertical):
            for suffix in ("", "_reverse"):
                wi = getattr(lstm, f"weight_ih_l0{suffix}")
                wh = getattr(lstm, f"weight_hh_l0{suffix}")
                bi = getattr(lstm, f"bias_ih_l0{suffix}")
                bh = getattr(lstm, f"bias_hh_l0{suffix}")
                nn.init.xavier_uniform_(wi)
                nn.init.orthogonal_(wh)
                nn.init.zeros_(bi); nn.init.zeros_(bh)
                bi.data[32:64] = 1.0  # Keras unit_forget_bias=True.
                # Keras has one bias vector. PyTorch sums two; fixing the second
                # at zero is functionally identical and preserves trainable count.
                bh.requires_grad_(False)

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.c1(x); p1 = self.pool(c1)
        c2 = self.c2(p1); p2 = self.pool(c2)
        c3 = self.c3(p2); p3 = self.pool(c3)
        c4 = self.c4(p3); p4 = self.pool(c4)
        n, _, h, w = p4.shape

        nhwc = p4.permute(0, 2, 3, 1).contiguous()
        horizontal, _ = self.horizontal(nhwc.reshape(n, h * w, 128))
        vertical_in = nhwc.transpose(1, 2).contiguous().reshape(n, h * w, 128)
        vertical, _ = self.vertical(vertical_in)
        horizontal = horizontal.reshape(n, h, w, 64).permute(0, 3, 1, 2).contiguous()
        vertical = vertical.reshape(n, w, h, 64).transpose(1, 2).permute(0, 3, 1, 2).contiguous()
        features = torch.cat((self.h_conv(horizontal), self.v_conv(vertical)), dim=1)

        c5 = self.c5(torch.cat((self.up5(features), c4), dim=1))
        c6 = self.c6(torch.cat((self.up6(c5), c3), dim=1))
        c7 = self.c7(torch.cat((self.up7(c6), c2), dim=1))
        c8 = self.c8(torch.cat((self.up8(c7), c1), dim=1))
        return self.out(c8)


def tv_loss_plus_var_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """Exact NCHW form of the official Keras loss."""
    target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_y = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_x = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    pred_y = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    total_variation = (target_x - pred_x).abs().mean() + (target_y - pred_y).abs().mean()
    error = prediction - target
    variance = (error.square().mean(dim=(1, 2, 3)) - error.mean(dim=(1, 2, 3)).square()).mean()
    return variance + 0.1 * total_variation
