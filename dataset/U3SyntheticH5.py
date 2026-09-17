"""Lazy, multi-worker-safe reader for the GFS/RME/RBR HDF5 datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import h5py
import torch
from torch.utils.data import Dataset


class U3SyntheticH5(Dataset):
    def __init__(self, root, split="train", test_snr=30,
                 transform: Callable | None=None, target_transform: Callable | None=None,
                 joint_transform: Callable | None=None, **_):
        self.root=Path(root); self.split=split; self.test_snr=int(test_snr)
        self.path=self.root/("train.h5" if split=="train" else f"test_{self.test_snr}dB.h5")
        if not self.path.exists(): raise FileNotFoundError(self.path)
        with h5py.File(self.path,"r") as f:
            self.length=len(f["psi"]); self.image_size=int(f.attrs["image_size"])
            if f["psi"].shape != f["phi"].shape: raise ValueError(f"shape mismatch in {self.path}")
        self.transform=transform; self.target_transform=target_transform; self.joint_transform=joint_transform
        self._file=None

    def __len__(self): return self.length

    def _handle(self):
        if self._file is None: self._file=h5py.File(self.path,"r")
        return self._file

    def __getitem__(self,index):
        f=self._handle(); wrapped=torch.from_numpy(f["psi"][index]).unsqueeze(0)
        unwrapped=torch.from_numpy(f["phi"][index]).unsqueeze(0); snr=torch.tensor([f["snr"][index]],dtype=torch.float32)
        if self.joint_transform: wrapped,unwrapped=self.joint_transform(wrapped,unwrapped)
        if self.transform: wrapped=self.transform(wrapped)
        if self.target_transform: unwrapped=self.target_transform(unwrapped)
        sample={"wrapped":wrapped,"unwrapped":unwrapped,"snr":snr,
                "wrapped_neg_norm":torch.clamp(wrapped/torch.pi,-1,1),
                "unwrapped_neg_norm":torch.clamp(unwrapped/(14*torch.pi),-1,1),
                "wrapped_cond":torch.cat((torch.sin(wrapped),torch.cos(wrapped)),dim=0)}
        for name in ("phi_wrapped_clean","coherence","dem","deformation_phase","noise_map"):
            if name in f: sample[name]=torch.from_numpy(f[name][index].astype("float32")).unsqueeze(0)
        for name in ("scene_id","region_id","partition","difficulty","deformation_type","scene_mode","satellite_id"):
            if name in f: sample[name]=torch.tensor(int(f[name][index]),dtype=torch.long)
        return sample

    def __getstate__(self):
        state=self.__dict__.copy(); state["_file"]=None; return state

    def __del__(self):
        if getattr(self,"_file",None) is not None: self._file.close()
