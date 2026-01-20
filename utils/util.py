from collections import defaultdict

import torch

import torch.nn.functional as F

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap continuous phase to [-pi, pi]."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))

    # psi = (phi + torch.pi) % (2 * torch.pi) - torch.pi
    #
    # return psi


def median_filter2d(input, kernel_size=3):
    # 确保 kernel_size 为奇数
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    padding = kernel_size // 2

    # 对每个通道和每个批次进行处理
    # 使用滑动窗口计算局部区域的中位数
    unfolded = F.unfold(input, kernel_size=kernel_size, padding=padding)
    unfolded = unfolded.view(input.size(0), input.size(1), kernel_size * kernel_size, -1)

    # 对每个窗口进行排序，并取中位数
    unfolded, _ = unfolded.sort(dim=2)
    median = unfolded[:, :, kernel_size * kernel_size // 2, :].view_as(input)

    return median



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.sums[k] += v
            self.counts[k] += 1

    def avg(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}


import torch.nn as nn
# import torch.nn.functional as F
class ExtraTokenCondition(nn.Module):
    """
    将 aux UNet 的 feature maps 转为 cross-attention tokens
    """
    def __init__(self,config, in_channels_list, cross_attention_dim):
        super().__init__()
        self.device = config.training.device


        self.proj_layers = nn.ModuleList([
            nn.Linear(c, cross_attention_dim) for c in in_channels_list
        ])

        # gate：非常重要，保证训练稳定
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, encoder_hidden_states, feats_dict):
        """
        encoder_hidden_states: [B, N, D] or None
        feats_dict: {"down3": [B,C,H,W], "mid": ...}
        """
        tokens_all = []

        for i, feat in enumerate(feats_dict.values()):
            # 下采样，控制 token 数
            feat = F.avg_pool2d(feat, kernel_size=2)

            B, C, H, W = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)   # [B, HW, C]

            assert tokens.shape[-1] == self.proj_layers[i].in_features, \
                f"Token dim {tokens.shape[-1]} != Linear in_features {self.proj_layers[i].in_features}"

            tokens = self.proj_layers[i](tokens)       # [B, HW, D]
            tokens_all.append(tokens)

        aux_tokens = torch.cat(tokens_all, dim=1)
        aux_tokens = torch.sigmoid(self.gate) * aux_tokens

        if encoder_hidden_states is None:
            return aux_tokens

        return torch.cat([encoder_hidden_states, aux_tokens], dim=1)


class UNetFeatureHook:
    def __init__(self, unet, hook_layers):
        """
        unet: 你的 UNet 实例
        hook_layers: dict(name -> module)
        """
        self.features = {}
        self.handles = []

        for name, module in hook_layers.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # output 可能是 tuple（你的 DownB 很可能是）
            if isinstance(output, tuple):
                self.features[name] = output[0]  # 主分支特征
            else:
                self.features[name] = output
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

