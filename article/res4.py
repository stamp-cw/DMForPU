import os.path

import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

from utils.util import multi_scale_wavelet

wrapped_mat_path = r"data/wrapped/wrapped.mat"
wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
wrapped_tensor = torch.from_numpy(wrapped_mat).float()

fig_size_W = 3.5
fig_size_H = 2.5
fig_dpi = 600

level = 3

img_path_dir = "res/res4"

img = np.sin(wrapped_mat)
img_path = f"{img_path_dir}/sin.png"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="twilight")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

img = np.cos(wrapped_mat)
img_path = f"{img_path_dir}/cos.png"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="twilight")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

sin_wrapped = multi_scale_wavelet(torch.sin(wrapped_tensor), 'db4', level=level)
sin_wrapped = sin_wrapped.unsqueeze(0)
b,c,h,w = sin_wrapped.shape

for i in range(c):
    img_array = sin_wrapped[:,i,:,:]
    img = img_array.squeeze()  # [H, W]
    img = img.detach().cpu().numpy()
    img_path = f"{img_path_dir}/sin_ms_{i}.png"
    plt.figure(figsize=(fig_size_W, fig_size_H))
    plt.imshow(img, cmap="twilight")
    plt.axis("off")
    plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

cos_wrapped = multi_scale_wavelet(torch.cos(wrapped_tensor), 'db4', level=level)
cos_wrapped = cos_wrapped.unsqueeze(0)
b,c,h,w =cos_wrapped.shape

for i in range(c):
    img_array = cos_wrapped[:,i,:,:]
    img = img_array.squeeze()  # [H, W]
    img = img.detach().cpu().numpy()
    img_path = f"{img_path_dir}/cos_ms_{i}.png"
    plt.figure(figsize=(fig_size_W, fig_size_H))
    plt.imshow(img, cmap="twilight")
    plt.axis("off")
    plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
    plt.close()