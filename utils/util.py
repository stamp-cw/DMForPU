from collections import defaultdict

import torch

import torch.nn.functional as F

import argparse

from scipy.fft import dctn, idctn
import numpy as np
import pywt

def multi_scale_wavelet(phase, wavelet, level):
    # phase: [1, H, W]
    p = phase.squeeze(0).detach().cpu().numpy()
    coeffs = wavelet_multiscale_decompose(p, wavelet, level)
    concat_list = []
    concat_list.append(F.interpolate(torch.as_tensor(coeffs[0], device=phase.device).unsqueeze(0).unsqueeze(0), size=phase.shape[-2:],mode='bilinear',align_corners=False).squeeze(0))

    for l in range(1, level+1):
        cH, cV, cD = coeffs[l]
        for c in (cH, cV, cD):
            t = torch.as_tensor(c, device=phase.device).unsqueeze(0).unsqueeze(0)
            concat_list.append(F.interpolate(t, size=phase.shape[-2:],mode='bilinear',align_corners=False).squeeze(0))
    return torch.concat(concat_list, dim=0)

def wavelet_multiscale_decompose(
        phase: np.ndarray,
        wavelet: str = 'db4',
        level: int = 3
):
    """
    单通道相位矩阵的小波多尺度分解

    Args:
        phase (np.ndarray): shape (H, W)，单通道相位矩阵
        wavelet (str): 小波基，如 'db4', 'haar', 'sym5'
        level (int): 分解层数

    Returns:
        coeffs: list
            coeffs[0]              -> cA_n（最粗尺度低频）
            coeffs[1:] 每层为 (cH, cV, cD)
    """
    if phase.ndim != 2:
        raise ValueError("phase 必须是二维单通道矩阵")

    coeffs = pywt.wavedec2(
        data=phase,
        wavelet=wavelet,
        level=level
    )
    return coeffs


import torch
import torch.nn.functional as F
import pywt

class MultiScaleWavelet:
    def __init__(self, wavelet='db4', level=3):
        """
        多尺度小波分解（Torch + PyWavelets）

        Args:
            wavelet (str): 小波类型，如 'db4', 'haar'
            level (int): 分解层数
        """
        self.wavelet = wavelet
        self.level = level

    def decompose(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 做多尺度小波分解并 concat 为多通道张量

        Args:
            x (torch.Tensor): [B,C,H,W] 单通道或多通道

        Returns:
            torch.Tensor: [B, C*(1+3*level), H, W] concat 后的多尺度系数
        """
        B, C, H, W = x.shape
        device = x.device
        out_list = []

        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                # 单通道分解
                img = x[b, c]  # [H, W]
                coeffs = pywt.wavedec2(img.cpu().numpy(), wavelet=self.wavelet, level=self.level)
                # coeffs[0] -> 最低频
                cA_n = torch.tensor(coeffs[0], device=device, dtype=x.dtype).unsqueeze(0)  # [1, h, w]
                # 上采样到原图大小
                cA_n = F.interpolate(cA_n.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                batch_coeffs.append(cA_n)

                # 每层细节系数
                for l in range(1, self.level+1):
                    cH, cV, cD = coeffs[l]
                    for cD_comp in (cH, cV, cD):
                        t = torch.tensor(cD_comp, device=device, dtype=x.dtype).unsqueeze(0)
                        t = F.interpolate(t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                        batch_coeffs.append(t)

            # concat channels
            out_list.append(torch.cat(batch_coeffs, dim=0))  # [C*(1+3*level), H, W]

        # Batch concat
        out = torch.stack(out_list, dim=0)  # [B, C*(1+3*level), H, W]
        return out
# def poisson_reconstruct_phase_fft_torch(gx, gy):
#     """
#     Reconstruct phase from gradients using FFT-based Poisson solver
#     (periodic boundary condition).
#
#     Args:
#         gx: torch.Tensor, (B, 1, H, W-1)
#         gy: torch.Tensor, (B, 1, H-1, W)
#
#     Returns:
#         phase: torch.Tensor, (B, 1, H, W)
#     """
#     # 1. divergence
#     # div = gradient_divergence_torch(gx, gy)  # (B,1,H,W)
#     div = gx  + gy                            # (B,1,H,W)
#     div = div.squeeze(1)                     # (B,H,W)
#
#     B, H, W = div.shape
#     device = div.device
#
#     # 2. FFT of divergence
#     div_hat = torch.fft.fft2(div)
#
#     # 3. Laplacian eigenvalues (periodic)
#     ky = torch.fft.fftfreq(H, device=device).view(-1, 1)
#     kx = torch.fft.fftfreq(W, device=device).view(1, -1)
#
#     denom = (
#             2 * torch.cos(2 * torch.pi * kx) +
#             2 * torch.cos(2 * torch.pi * ky) -
#             4
#     )
#
#     denom[0, 0] = 1.0  # avoid division by zero (DC component)
#
#     # 4. Solve in Fourier domain
#     phase_hat = div_hat / denom
#
#     phase_hat[:, 0, 0] = 0.0  # fix global constant (mean = 0)
#
#     # 5. inverse FFT
#     phase = torch.fft.ifft2(phase_hat).real
#
#     return phase.unsqueeze(1)  # (B,1,H,W)


def poisson_reconstruct_phase(gx_tensor, gy_tensor):
    gx = gx_tensor.cpu().detach().numpy()
    gy = gy_tensor.cpu().detach().numpy()

    b, c, h, w = gx.shape

    # Compute the divergence（保持与你原始代码一致的差分风格）
    # div = torch.zeros((h, w), dtype=np.float64)
    # div[:, :-1] += gx[:, :-1]
    # div[:, 1:]  -= gx[:, :-1]
    # div[:-1, :] += gy[:-1, :]
    # div[1:,  :] -= gy[:-1, :]

    div = gx + gy

    # Solve Poisson equation using DCT (Neumann BC)
    b = dctn(div, type=2, norm='ortho')

    ky = np.arange(h).reshape(h, 1)
    kx = np.arange(w).reshape(1, w)
    denom = (2 * np.cos(np.pi * kx / w) - 2) + (2 * np.cos(np.pi * ky / h) - 2)

    # Neumann Laplacian has a zero eigenvalue at (0,0): fix gauge by enforcing zero-mean
    denom[0, 0] = 1.0
    f_hat = b / denom
    f_hat[0, 0] = 0.0

    f = idctn(f_hat, type=2, norm='ortho')

    # f_tensor = torch.as_tensor(f,device=gx_tensor.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    f_tensor = torch.as_tensor(f,device=gx_tensor.device) # (1,1,H,W)
    # return f.astype(np.float32)
    return f_tensor



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            # 自动尝试把字符串转成数字
            if isinstance(value, str):
                try:
                    if value.lower() in ["true", "false"]:   # 转成 bool
                        new_value = value.lower() == "true"
                    elif "." in value or "e" in value.lower():  # float
                        new_value = float(value)
                    else:  # int
                        new_value = int(value)
                except Exception:
                    new_value = value  # 转换失败就保持字符串
            else:
                new_value = value

        setattr(namespace, key, new_value)
    return namespace

def update_dict(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            update_dict(base[k], v)
        else:
            base[k] = v
    return base

def unflatten_dict(flat_dict, sep="."):
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return result

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
        # print(self.counts)
        # print(metrics)
        # print(self.sums)

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


from diffusers.models.attention import Attention

class AttentionKVInjector:
    def __init__(self, unet):
        self.handles = []
        self.kv_cache = None  # (k, v)

        for module in unet.modules():
            if isinstance(module, Attention):
                h = module.register_forward_hook(self._hook_fn)
                self.handles.append(h)

    def set_kv(self, k, v):
        """
        k, v: [B, N, D]
        """
        self.kv_cache = (k, v)

    def _hook_fn(self, module, inputs, output):
        """
        inputs:
          hidden_states,
          encoder_hidden_states,
          attention_mask,
          ...
        """
        if self.kv_cache is None:
            return output

        # 判断是不是 cross-attention
        encoder_hidden_states = inputs[1] if len(inputs) > 1 else None
        if encoder_hidden_states is None:
            return output  # skip self-attn

        hidden_states = inputs[0]
        k_ext, v_ext = self.kv_cache

        k_ext = k_ext.to(hidden_states.device)
        v_ext = v_ext.to(hidden_states.device)

        # ===== 原 Attention 内部逻辑（复刻）=====
        query = module.to_q(hidden_states)

        key   = module.to_k(k_ext)
        value = module.to_v(v_ext)

        query = module.head_to_batch_dim(query)
        key   = module.head_to_batch_dim(key)
        value = module.head_to_batch_dim(value)

        attn_scores = torch.baddbmm(
            torch.empty(
                query.shape[0], query.shape[1], key.shape[1],
                device=query.device,
                dtype=query.dtype,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=module.scale,
        )

        attn_probs = attn_scores.softmax(dim=-1)
        hidden_states = torch.bmm(attn_probs, value)

        hidden_states = module.batch_to_head_dim(hidden_states)
        hidden_states = module.to_out[0](hidden_states)
        hidden_states = module.to_out[1](hidden_states)

        return hidden_states

    def remove(self):
        for h in self.handles:
            h.remove()

