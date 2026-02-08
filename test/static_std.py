import os

import torch
import tqdm

from dataset.InSARDLPUMat import InSARDLPUMat
from dataset.SyntheticPUMat import SyntheticPUMat

# from dataset.SyntheticPUMatCutMid import SyntheticPUMatCutMid
# from dataset.SyntheticPUMatMid import SyntheticPUMatMid

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# train_dataset = SyntheticPUMatMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mid', split='train')
# train_dataset = SyntheticPUMatMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mid', split='test')
# train_dataset = SyntheticPUMatMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128MidTest', split='test')
# train_dataset = SyntheticPUMatCutMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Wav', split='train')
# train_dataset = SyntheticPUMatCutMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Big', split='train')
# train_dataset = InSARDLPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat256Big', split='train')
# test_dataset = InSARDLPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat256Big', split='test')
# all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
# train_dataset = SyntheticPUMatCutMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mid', split='train')
# train_dataset = SyntheticPUMatCutMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMatCut32MidTest', split='train')

# train_dataset = SyntheticPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mch', split='train')
# test_dataset = SyntheticPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mch', split='test')

train_dataset = SyntheticPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Big', split='train')
test_dataset = SyntheticPUMat(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Big', split='test')

all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

data_loader = torch.utils.data.DataLoader(
    # train_dataset,
    # test_dataset,
    all_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    drop_last=False
)

dataloader = data_loader

pixel_sum = torch.zeros(1, dtype=torch.float64)
pixel_sq_sum = torch.zeros(1, dtype=torch.float64)
pixel_count = 0

# pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.end_epoch}")
a_max = -float('inf')
a_min = float('inf')
for batch in tqdm.tqdm(dataloader):
    x = batch['unwrapped'].double()
    pixel_sum += x.sum()
    pixel_sq_sum += (x ** 2).sum()
    pixel_count += x.numel()
    b_max = x.max()
    b_min = x.min()
    if b_max > a_max:
        a_max = b_max
    if b_min < a_min:
        a_min = b_min
mean = (pixel_sum / pixel_count).item()
std = ((pixel_sq_sum / pixel_count - mean ** 2) ** 0.5).item()
print(f"Mean: {mean}, Std: {std}")
print(f"Max: {a_max}, Min: {a_min}")
print(f"Max / 2*pi: {a_max /(2 * torch.pi)}, Min / 2*pi: {a_min/(2 * torch.pi)}")
print(f"(Max + pi) / 2*pi: {(a_max + torch.pi) /(2 * torch.pi)}, Min / 2*pi: {a_min/(2 * torch.pi)}")
