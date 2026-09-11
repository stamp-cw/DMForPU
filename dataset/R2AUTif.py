# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import re


class R2AUTif(Dataset):
    """
    InSAR 数据集（TIF）
    - noisy_phase: 缠绕相位
    - unwrapped_phase: 真实相位
    - 按 idx + 固定随机种子划分 train/test
    """

    def __init__(
        self,
        root,
        split='train',   # 'train' or 'test'
        seed=42,         # 固定随机种子
        train_ratio=0.9,
        transform=None,
        target_transform=None,
        joint_transform=None,
        k_min=0,
        k_max=3,
        wavelet_level=3,
        wavelet_type='db4',
        mean=10.01,
        std=5.74,
        scale_alpha=2,
    ):
        super().__init__()

        assert split in ['train', 'test']

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.scale_k = k_max - k_min

        self.k_min = k_min
        self.k_max = k_max

        root = Path(root)

        self.noisy_dir = root / "noisy_phase"
        self.unwrapped_dir = root / "unwrapped_phase"

        # =========================
        # 1️⃣ 建立配对文件
        # =========================
        # noisy_files = {p.stem: p for p in self.noisy_dir.glob("*.tif")}
        # unwrapped_files = {p.stem: p for p in self.unwrapped_dir.glob("*.tif")}

        noisy_files = {
            self.extract_id(p.stem): p
            for p in self.noisy_dir.glob("*.tif")
        }

        unwrapped_files = {
            self.extract_id(p.stem): p
            for p in self.unwrapped_dir.glob("*.tif")
        }

        keys = sorted(set(noisy_files) & set(unwrapped_files))
        if len(keys) == 0:
            raise RuntimeError("没有匹配的 tif 文件")

        # =========================
        # 2️⃣ 固定随机划分
        # =========================
        rng = np.random.RandomState(seed)
        indices = np.arange(len(keys))
        rng.shuffle(indices)

        split_idx = int(len(indices) * train_ratio)

        if split == 'train':
            selected_idx = indices[:split_idx]
        else:
            selected_idx = indices[split_idx:]

        self.keys = [keys[i] for i in selected_idx]

        self.paths = {
            "noisy": noisy_files,
            "unwrapped": unwrapped_files,
        }

    def extract_id(self, name):
        """
        从文件名提取编号，例如：
        noisy_phase_00001 -> 00001
        """
        match = re.search(r'(\d+)$', name)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"无法解析文件名: {name}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        key = self.keys[idx]

        # =========================
        # 3️⃣ 读取 tif
        # =========================
        noisy = tiff.imread(str(self.paths["noisy"][key]))
        unwrapped = tiff.imread(str(self.paths["unwrapped"][key]))

        # 转 tensor
        noisy = torch.from_numpy(noisy).float()
        unwrapped = torch.from_numpy(unwrapped).float()

        # 保证 shape = [1, H, W]
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(0)
        if unwrapped.ndim == 2:
            unwrapped = unwrapped.unsqueeze(0)

        # =========================
        # transform
        # =========================
        if self.joint_transform:
            noisy, unwrapped = self.joint_transform(noisy, unwrapped)
        if self.transform:
            noisy = self.transform(noisy)
        if self.target_transform:
            unwrapped = self.target_transform(unwrapped)

        # =========================
        # 归一化（和你原来一致）
        # =========================
        wrapped_neg_norm = torch.clamp(noisy / torch.pi, -1, 1)
        unwrapped_norm = (unwrapped - self.k_min * 2 * torch.pi) / ( (self.k_max - self.k_min) * 2 * torch.pi)
        unwrapped_norm = torch.clamp(unwrapped_norm, 0, 1)
        unwrapped_neg_norm = unwrapped_norm * 2 - 1

        # wrapped_cond = noisy / torch.pi
        wrapped = noisy
        sin_wrapped = torch.sin(wrapped)
        cos_wrapped = torch.cos(wrapped)
        wrapped_cond = torch.cat([sin_wrapped, cos_wrapped], dim=0)

        return {
            "wrapped": noisy,
            "unwrapped": unwrapped,
            "wrapped_neg_norm": wrapped_neg_norm,
            "unwrapped_neg_norm": unwrapped_neg_norm,
            "wrapped_cond": wrapped_cond,
        }