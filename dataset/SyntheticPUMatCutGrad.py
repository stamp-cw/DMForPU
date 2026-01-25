# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SyntheticPUMatCutGrad(Dataset):
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

        self.scale_k = k_max - k_min
        self.k_min = k_min
        self.k_max = k_max


        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.data_root = Path(root)

        self.wrapped_dir = self.data_root / f"{split}_in"
        self.unwrapped_dir = self.data_root / f"{split}_gt"

        # print(os.listdir(self.wrapped_dir))

        wrapped_files = {p.stem: p for p in self.wrapped_dir.glob("*.mat")}
        unwrapped_files = {p.stem: p for p in self.unwrapped_dir.glob("*.mat")}
        # print(wrapped_files)
        # print(unwrapped_files)
        self.keys = sorted(set(wrapped_files).intersection(unwrapped_files))
        if not self.keys:
            raise FileNotFoundError(f"No paired files found under {self.data_root} for split {split}")

        self.paths = {
            "wrapped": wrapped_files,
            "unwrapped": unwrapped_files,
        }

    def __len__(self) -> int:
        return len(self.keys)

    def phase_gradient_torch(self, phase):
        """
        Args:
            phase: torch.Tensor, shape (H, W) or (B, 1, H, W)

        Returns:
            gx, gy: real-valued gradients
        """

        # gx = phase[..., :, 1:] - phase[..., :, :-1]
        # gy = phase[..., 1:, :] - phase[..., :-1, :]
        phase.unsqueeze(0)

        kx = torch.tensor(
            [[0, 0, 0],
             [-1, 1, 0],
             [0, 0, 0]],
            device=phase.device, dtype=phase.dtype
        ).view(1, 1, 3, 3)

        gx = F.conv2d(phase, kx, padding=1, groups=1, stride=1)
        gx[..., :, 0] = gx[..., :, 1] - (gx[..., :, 2] - gx[..., :, 1])

        ky = torch.tensor(
            [[0, -1, 0],
             [0,  1, 0],
             [0,  0, 0]],
            device=phase.device, dtype=phase.dtype
        ).view(1, 1, 3, 3)
        gy = F.conv2d(phase, ky, padding=1, groups=1, stride=1)
        gy[..., 0, :] = gy[..., 1, :] - (gy[..., 2, :] - gy[..., 1, :])

        return gx, gy

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.keys[idx]
        wrapped = sio.loadmat(str(self.paths["wrapped"][key]))['input']
        unwrapped = sio.loadmat(str(self.paths["unwrapped"][key]))['gt']
        wrapped = torch.as_tensor(wrapped).unsqueeze(0)
        unwrapped = torch.as_tensor(unwrapped).unsqueeze(0)

        unwrapped = unwrapped - (torch.min(unwrapped) // (torch.pi * 2)) * (torch.pi * 2)
        unwrapped = torch.clamp(unwrapped, min=0.0) # [0, inf]


        if self.joint_transform:
            wrapped, unwrapped = self.joint_transform(wrapped, unwrapped)
        if self.transform:
            wrapped = self.transform(wrapped)
        if self.target_transform:
            unwrapped = self.target_transform(unwrapped)

        wrapped_neg_norm = wrapped / torch.pi
        wrapped_neg_norm = torch.clamp(wrapped_neg_norm, -1, 1)

        unwrapped_norm = unwrapped / (2 * torch.pi * self.scale_k)
        unwrapped_norm = torch.clamp(unwrapped_norm, 0, 1)
        unwrapped_neg_norm = unwrapped_norm * 2 - 1

        unwrapped_grad_x , unwrapped_grad_y = self.phase_gradient_torch(unwrapped_neg_norm) # [-2,2]
        # unwrapped_grad_x_neg_norm = unwrapped_grad_x / 2 # [-1,1]
        # unwrapped_grad_y_neg_norm = unwrapped_grad_y / 2 # [-1,1]
        unwrapped_grad_neg_norm = torch.cat([unwrapped_grad_x, unwrapped_grad_y], dim=0) * 8
        # unwrapped_grad_neg_norm = torch.cat([unwrapped_grad_x, unwrapped_grad_y], dim=0)
        # print(f"unwrapped_grad_x shape: {unwrapped_grad_x.shape}, unwrapped_grad_y shape: {unwrapped_grad_y.shape}")
        # unwrapped_grad_neg_norm = torch.cat([unwrapped_grad_x, unwrapped_grad_y], dim=0) / 2

        wrapped_cond = torch.cat([torch.sin(wrapped), torch.cos(wrapped)], dim=0)

        sample = {
            "wrapped": wrapped,
            "unwrapped": unwrapped,
            "wrapped_neg_norm": wrapped_neg_norm,
            "unwrapped_neg_norm": unwrapped_neg_norm,
            "wrapped_cond": wrapped_cond,
            "unwrapped_grad_neg_norm": unwrapped_grad_neg_norm
        }
        return sample