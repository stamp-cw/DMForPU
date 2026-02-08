# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class SyntheticData(Dataset):
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
        scale_k=5
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

        self.K = scale_k

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.data_root = Path(root)

        self.wrapped_dir = self.data_root / f"{split}_wrapped"
        self.unwrapped_dir = self.data_root / f"{split}_unwrapped"

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.keys[idx]
        wrapped = sio.loadmat(str(self.paths["wrapped"][key]))['wrapped']
        unwrapped = sio.loadmat(str(self.paths["unwrapped"][key]))['unwrapped']
        wrapped = torch.as_tensor(wrapped).unsqueeze(0)
        unwrapped = torch.as_tensor(unwrapped).unsqueeze(0)

        if self.joint_transform:
            wrapped, unwrapped = self.joint_transform(wrapped, unwrapped)
        if self.transform:
            wrapped = self.transform(wrapped)
        if self.target_transform:
            unwrapped = self.target_transform(unwrapped)

        K = self.K # k range 3-5
        # wrapped = wrapped + torch.pi
        # wrapped_norm = wrapped / torch.pi
        unwrapped_neg_norm = unwrapped / (torch.pi * K)
        unwrapped_neg_norm = torch.clamp(unwrapped_neg_norm, -1, 1)
        unwrapped_norm =  (unwrapped_neg_norm + 1) / 2
        # wrapped_cond = torch.stack([torch.sin(wrapped), torch.cos(wrapped)], dim=0)
        # wrapped_cond = wrapped / (2 * torch.pi)
        wrapped_cond = wrapped / torch.pi

        unwrapped_sub_wrapped = unwrapped - wrapped
        unwrapped_sub_wrapped_norm = unwrapped_sub_wrapped / (torch.pi * K)
        unwrapped_sub_wrapped_norm = torch.clamp(unwrapped_sub_wrapped_norm, 0, 1)

        sample = {
            "wrapped": wrapped,
            # "wrapped_fp16": wrapped.to(torch.float16),
            "unwrapped": unwrapped,
            # "unwrapped_fp16": unwrapped.to(torch.float16),
            # "wrapped_norm": wrapped_norm,
            # "wrapped_norm_fp16": wrapped_norm.to(torch.float16),
            "unwrapped_norm": unwrapped_norm,
            "unwrapped_neg_norm": unwrapped_neg_norm,
            "wrapped_cond": wrapped_cond,
            # "wrapped_cond_fp16": wrapped_cond.to(torch.float16),
            "unwrapped_sub_wrapped": unwrapped_sub_wrapped,
            # "unwrapped_sub_wrapped_fp16": unwrapped_sub_wrapped.to(torch.float16),
            "unwrapped_sub_wrapped_norm": unwrapped_sub_wrapped_norm,
            # "unwrapped_sub_wrapped_norm_fp16": unwrapped_sub_wrapped_norm.to(torch.float16)
        }
        return sample