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

        self.mode = split
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
        复现 Unsupervised-PU：在弧度域生成零均值高斯扰动，并返回
        重缠绕后相对于原相位的有效扰动。
        """
        snr_db = torch.as_tensor(SNR, dtype=wrapped.dtype, device=wrapped.device)
        signal_power = torch.pow(wrapped.new_tensor(10.0), wrapped.new_tensor(0.1))
        requested_snr = torch.pow(wrapped.new_tensor(10.0), snr_db / 10)
        noise_std = torch.sqrt(signal_power / requested_snr)
        phase_noise = noise_std * torch.randn_like(wrapped)
        return self.wrap_phase(wrapped + phase_noise) - wrapped

    def wrap_phase(self, phi: torch.Tensor) -> torch.Tensor:
        """Wrap continuous phase to [-pi, pi]."""
        return torch.atan2(torch.sin(phi), torch.cos(phi))

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
        if self.mode == 'test':
            sample_snr = torch.as_tensor(self.snr).float().reshape(())
            wrapped_noise = self.get_Gaussian_Noise(wrapped, sample_snr)
        else:
            db_lst = torch.tensor([0, 5, 10, 20, 30])
            # db_lst = torch.tensor([20, 100])
            # db_lst = torch.tensor([100])
            # self.snr = torch.FloatTensor(1).uniform_(0, self.snr).item()
            sample_snr = db_lst[torch.randint(0, len(db_lst), ())].float()
            wrapped_noise = self.get_Gaussian_Noise(wrapped, sample_snr)

        # wrapped_noise = self.get_Gaussian_Noise(wrapped, self.snr)

        wrapped_noisy = wrapped + wrapped_noise
        # wrapped_noisy = wrapped
        # wrapped_noisy = torch.clamp(wrapped_noisy, -torch.pi, torch.pi)
        wrapped_noisy = self.wrap_phase(wrapped_noisy)
        wrapped_neg_norm = wrapped_noisy / torch.pi

        # neg_norm_diffusion
        unwrapped_norm = unwrapped / (2 * torch.pi * self.scale_k)
        # unwrapped_norm = (unwrapped + torch.pi) / (2 * torch.pi * self.scale_k)
        unwrapped_norm = torch.clamp(unwrapped_norm, 0, 1)
        unwrapped_neg_norm = unwrapped_norm * 2 - 1

        # wrapped_cond
        # sin_wrapped = multi_scale_wavelet(torch.sin(wrapped), self.wavelet_type, level=self.wavelet_level)
        # cos_wrapped = multi_scale_wavelet(torch.cos(wrapped), self.wavelet_type, level=self.wavelet_level)
        # wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        # sin_wrapped = multi_scale_wavelet(torch.sin(wrapped_noise), self.wavelet_type, level=self.wavelet_level)
        # cos_wrapped = multi_scale_wavelet(torch.cos(wrapped_noise), self.wavelet_type, level=self.wavelet_level)
        # wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        sin_wrapped = torch.sin(wrapped_noisy)
        cos_wrapped = torch.cos(wrapped_noisy)
        wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        sample = {
            # "wrapped": wrapped,
            "wrapped":  wrapped_noisy,
            "unwrapped": unwrapped,
            "unwrapped_neg_norm": unwrapped_neg_norm,
            "wrapped_neg_norm": wrapped_neg_norm,
            "wrapped_cond": wrapped_cond,
            "wrapped_noise": wrapped_noise,
            "snr_db": sample_snr,
        }
        return sample
