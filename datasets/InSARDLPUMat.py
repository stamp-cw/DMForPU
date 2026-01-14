# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict
import scipy.io as sio
import torch
from torch.utils.data import Dataset


class InSARDLPUMat(Dataset):
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
    ):
        """
        Args:
            root (str): 数据集根目录
            split (str): 'train' 或 'test'
            transform (callable, optional): 对图像的变换
            target_transform (callable, optional):
            joint_transform (callable, optional):
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        assert split in ['train', 'test'], "split 必须是 'train' 或 'test'"

        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        self.data_root = Path(root)

        self.wrapped_dir = self.data_root / f"{split}_wrapped"
        self.unwrapped_dir = self.data_root / f"{split}_absolute"

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
        wrapped = sio.loadmat(str(self.paths["wrapped"][key]))['input']
        unwrapped = sio.loadmat(str(self.paths["unwrapped"][key]))['output']
        wrapped = torch.as_tensor(wrapped).unsqueeze(0)
        unwrapped = torch.as_tensor(unwrapped).unsqueeze(0)

        if self.joint_transform:
            wrapped, unwrapped = self.joint_transform(wrapped, unwrapped)
        if self.transform:
            wrapped = self.transform(wrapped)
        if self.target_transform:
            unwrapped = self.target_transform(unwrapped)

        sample = {"wrapped": wrapped.float(), "unwrapped": unwrapped.float()}
        return sample