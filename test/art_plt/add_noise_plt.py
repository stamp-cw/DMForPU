import os.path

import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import tqdm
from diffusers import DDPMScheduler


gt_unwrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/article/data/gt/gt.mat')['gt']
gt_unwrapped = torch.as_tensor(gt_unwrapped).unsqueeze(0).unsqueeze(0)

noise = torch.randn_like(gt_unwrapped)
scheduler = DDPMScheduler(num_train_timesteps=1000)
for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
    noisy = scheduler.add_noise(gt_unwrapped, noise, t)
    img = noisy.squeeze()  # [H, W]
    img = img.detach().cpu().numpy()
    # img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
    img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/article/res/res3/noisy_t{t}.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="turbo")
    plt.axis("off")
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()