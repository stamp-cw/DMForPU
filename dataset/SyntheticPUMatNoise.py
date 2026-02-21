# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from utils.util import multi_scale_wavelet
from torch.distributions import Normal


class SyntheticPUMatNoise(Dataset):
    """
    InSARDLPU 数据集，Mat文件格式
    """

    def __init__(
        self,
        root,
        split='train',
        transform=None,
        target_transform=None,
        joint_transform=None,
        k_min = 0,
        k_max = 3,
        wavelet_level = 3,
        wavelet_type = 'db4',
        mean = 10.01,
        std = 5.74,
        scale_alpha = 2,
        snr = 0,
    ):
        """
        Args:
            root (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform (callable, optional): 对图像的变换
            target_transform (callable, optional):
            joint_transform (callable, optional):
        """
        # super().__init__(root, transform=transform, target_transform=target_transform)
        super().__init__()

        assert split in ['train', 'test'], "split 必须是 'train' 或 'test'"

        self.snr = snr
        self.mean = mean
        self.std = std
        self.normal = Normal(0.0, 1.0)
        self.scale_alpha = scale_alpha
        self.scale_k = k_max - k_min
        self.k_min = k_min
        self.k_max = k_max
        self.wavelet_level = wavelet_level
        self.wavelet_type = wavelet_type

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.data_root = Path(root)

        self.wrapped_dir = self.data_root / f"{split}_in"
        self.unwrapped_dir = self.data_root / f"{split}_gt"

        wrapped_files = {p.stem: p for p in self.wrapped_dir.glob("*.mat")}
        unwrapped_files = {p.stem: p for p in self.unwrapped_dir.glob("*.mat")}
        self.keys = sorted(set(wrapped_files).intersection(unwrapped_files))
        if not self.keys:
            raise FileNotFoundError(f"No paired files found under {self.data_root} for split {split}")

        self.paths = {
            "wrapped": wrapped_files,
            "unwrapped": unwrapped_files,
        }

    def __len__(self) -> int:
        return len(self.keys)

    def get_Gaussian_Noise(self, wrapped, SNR):
        """
        根据实际图像计算信号功率，生成指定SNR的高斯噪声
        """
        # 1. 归一化到[0,1]并计算信号功率
        # img_norm = image.astype(np.float64) / 255.0
        wrapped_norm = (wrapped + torch.pi) / (2 * torch.pi)
        # wrapped_norm = torch.clamp(wrapped_norm, 0, 1)

        sigPower = torch.mean(wrapped_norm ** 2)  # ✅ 从实际图像计算

        # 2. dB转线性，计算噪声功率和标准差
        reqSNR = 10 ** (SNR / 10)
        noisePower = sigPower / reqSNR
        std = torch.sqrt(noisePower)
        # 3. 生成噪声
        wrapped_norm_noise = std * torch.randn_like(wrapped, device=wrapped.device)
        wrapped_noise = wrapped_norm_noise * (2 * torch.pi) - torch.pi
        return wrapped_noise

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.keys[idx]
        wrapped = sio.loadmat(str(self.paths["wrapped"][key]))['input']
        unwrapped = sio.loadmat(str(self.paths["unwrapped"][key]))['gt']
        wrapped = torch.as_tensor(wrapped).unsqueeze(0)
        unwrapped = torch.as_tensor(unwrapped).unsqueeze(0)

        if self.joint_transform:
            wrapped, unwrapped = self.joint_transform(wrapped, unwrapped)
        if self.transform:
            wrapped = self.transform(wrapped)
        if self.target_transform:
            unwrapped = self.target_transform(unwrapped)

        # unwrapped = unwrapped - (torch.min(unwrapped) // (torch.pi * 2)) * (torch.pi * 2)
        # unwrapped = torch.clamp(unwrapped, min=0.0) # [0, inf]

        wrapped_neg_norm = wrapped / torch.pi
        # wrapped_neg_norm = torch.clamp(wrapped_neg_norm, -1, 1)
        wrapped_noise = self.get_Gaussian_Noise(wrapped, self.snr)

        # neg_norm_diffusion
        unwrapped_norm = unwrapped / (2 * torch.pi * self.scale_k)
        # unwrapped_norm = (unwrapped + torch.pi) / (2 * torch.pi * self.scale_k)
        # unwrapped_norm = torch.clamp(unwrapped_norm, 0, 1)
        unwrapped_neg_norm = unwrapped_norm * 2 - 1

        # wrapped_cond
        # sin_wrapped = multi_scale_wavelet(torch.sin(wrapped), self.wavelet_type, level=self.wavelet_level)
        # cos_wrapped = multi_scale_wavelet(torch.cos(wrapped), self.wavelet_type, level=self.wavelet_level)
        # wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        # sin_wrapped = multi_scale_wavelet(torch.sin(wrapped_noise), self.wavelet_type, level=self.wavelet_level)
        # cos_wrapped = multi_scale_wavelet(torch.cos(wrapped_noise), self.wavelet_type, level=self.wavelet_level)
        # wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        sin_wrapped = torch.sin(wrapped)
        cos_wrapped = torch.cos(wrapped)
        wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        sample = {
            # "wrapped": wrapped,
            "wrapped": wrapped,
            "unwrapped": unwrapped,
            "unwrapped_neg_norm": unwrapped_neg_norm,
            "wrapped_neg_norm": wrapped_neg_norm,
            "wrapped_cond": wrapped_cond,
            "wrapped_noise": wrapped_noise,
        }
        return sample