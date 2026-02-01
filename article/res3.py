import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import tqdm
from diffusers import DDPMScheduler


fig_size_W = 3.5
fig_size_H = 2.5
fig_dpi = 600

gt_mat_path = r"data/gt/gt.mat"
gt_mat = sio.loadmat(gt_mat_path)['gt']
gt_tensor = torch.from_numpy(gt_mat).unsqueeze(0).unsqueeze(0).float()

img_path_dir = r"res/res3"
noise = torch.randn_like(gt_tensor)
scheduler = DDPMScheduler(num_train_timesteps=1000)
for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
    noisy = scheduler.add_noise(gt_tensor, noise, t)
    img = noisy.squeeze()  # [H, W]
    img = img.detach().cpu().numpy()
    img_path = f"{img_path_dir}/noisy_t{t}.png"
    plt.figure(figsize=(fig_size_W, fig_size_H))
    plt.imshow(img, cmap="turbo")
    plt.axis("off")
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()