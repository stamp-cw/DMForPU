"""Official PUNet architecture for InSAR phase unwrapping.

Ported from Wu-Patrick/Deformation-Monitoring-Dev, commit
5be3c0117e20742bf4f68dc828fc0399d1e362b4, model/PUNet.py.
"""
import torch
import torch.nn as nn

from selector.model_selector import register_model


class DilatedBlock(nn.Module):
    """Three dilation branches followed by residual fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(out_channels * 3, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat((self.layer1(x), self.layer2(x), self.layer3(x)), dim=1)
        return self.relu(self.bn(x + self.conv(merged)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(x + self.conv(x)))


@register_model(name=["PUNet"])
class PUNet(nn.Module):
    """The official 64-channel, 8-dilated-block, 10-residual-block PUNet."""

    def __init__(self, config=None):
        super().__init__()
        output_channels = 1
        if config is not None and hasattr(config, "model"):
            output_channels = getattr(config.model, "out_channels", output_channels)

        self.inc = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(*[DilatedBlock(64, 64) for _ in range(8)])
        self.residual = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(10)])
        self.outc = nn.Conv2d(64, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        x = self.dilated(x)
        x = self.residual(x)
        return self.outc(x)
