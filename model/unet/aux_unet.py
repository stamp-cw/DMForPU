import torch
import torch.nn as nn
import torch.nn.functional as F

from selector.model_selector import register_model


@register_model(name='AuxUNet')
class AuxUNet(nn.Module):
    """
    条件 UNet：输入多频 wrapped phase latent
    输出多尺度 feature（不做 diffusion）
    """
    def __init__(self, config ,in_channels=1, base_channels=128):
        super().__init__()
        self.config = config

        self.down1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down2 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.down3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)

        self.mid = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)

    def forward(self, x):
        feats = {}

        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        feats["down2"] = x              # [B, 128, H/2, W/2]

        x = F.relu(self.down3(x))
        feats["down3"] = x              # [B, 256, H/4, W/4]

        x = F.relu(self.mid(x))
        feats["mid"] = x                # [B, 256, H/4, W/4]

        return feats
