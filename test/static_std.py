import os

import torch
import tqdm

from dataset.SyntheticPUMatMid import SyntheticPUMatMid

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_dataset = SyntheticPUMatMid(root='/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Mid', split='train')


data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=False
)

dataloader = data_loader

pixel_sum = torch.zeros(1, dtype=torch.float64)
pixel_sq_sum = torch.zeros(1, dtype=torch.float64)
pixel_count = 0

# pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.end_epoch}")

for batch in tqdm.tqdm(dataloader):
    x = batch['unwrapped'].double()
    pixel_sum += x.sum()
    pixel_sq_sum += (x ** 2).sum()
    pixel_count += x.numel()

mean = (pixel_sum / pixel_count).item()
std = ((pixel_sq_sum / pixel_count - mean ** 2) ** 0.5).item()
print(f"Mean: {mean}, Std: {std}")
