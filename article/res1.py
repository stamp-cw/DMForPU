# synpu与insar_dlpu，wwf-pu解缠效果图
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

from article.load_article_data import _to_numpy_2d, pi_formatter

#synpu
wrapped_mat_path = r"data/wrapped/wrapped.mat"
gt_mat_path = r"data/gt/gt.mat"

wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
gt_mat = sio.loadmat(gt_mat_path)['gt']

ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']
ours_mat = _to_numpy_2d(ours_pred)

#insar_dlpu
dlpu_wrapped_mat_path = r"data_dlpu/wrapped/wrapped.mat"
dlpu_gt_mat_path = r"data_dlpu/gt/gt.mat"
dlpu_wrapped_mat = sio.loadmat(dlpu_wrapped_mat_path)['input']
# dlpu_gt_mat = sio.loadmat(dlpu_gt_mat_path)['gt']
dlpu_gt_mat = sio.loadmat(dlpu_gt_mat_path)['output']

dlpu_ours_pred_batch_path = "data_dlpu/ours/samples_0_1.pt"
dlpu_ours_pred_batch_pt = torch.load(dlpu_ours_pred_batch_path)
dlpu_ours_pred = dlpu_ours_pred_batch_pt['pred_unwrapped']
dlpu_ours_mat = _to_numpy_2d(dlpu_ours_pred)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    # latex
    # "text.usetex": True,
    # "font.family": "serif"
})

# 单张图

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5

img = wrapped_mat
img_path = r"res/res1/wrapped.png"
pdf_img_path = r"res/res1/wrapped.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="twilight")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

img = dlpu_wrapped_mat
img_path = r"res/res1/dlpu_wrapped.png"
pdf_img_path = r"res/res1/dlpu_wrapped.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="twilight")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

img = gt_mat
img_path = r"res/res1/gt.png"
pdf_img_path = r"res/res1/gt.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="turbo")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

img = dlpu_gt_mat
img_path = r"res/res1/gt.png"
pdf_img_path = r"res/res1/gt.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="turbo")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()


img = ours_mat
img_path = r"res/res1_1/ours.png"
pdf_img_path = r"res/res1_1/ours.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="turbo")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()


img = dlpu_ours_mat
img_path = r"res/res1_1/ours.png"
pdf_img_path = r"res/res1_1/ours.pdf"
plt.figure(figsize=(fig_size_W, fig_size_H))
plt.imshow(img, cmap="turbo")
plt.axis("off")
plt.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.close()

#多图合一

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
png_img_path = r"res/res1/figure.png"
pdf_img_path = r"res/res1/figure.pdf"
# raw = 1
raw = 2
col = 3
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()

# raw 1, col 1
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
ax_0 = axes[0]
im_0 = ax_0.imshow(wrapped_mat, cmap="twilight", norm=color_norm)
ax_0.set_xlabel("(a) Wrapped", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_0, ax=ax_0, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

# raw 1, col 2
color_norm = colors.Normalize(vmin=0)
ax_1 = axes[1]
im_1 = ax_1.imshow(gt_mat, cmap="turbo", norm=color_norm)
ax_1.set_xlabel("(b) Gt", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_1, ax=ax_1, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

# raw 1, col 3
color_norm = colors.Normalize(vmin=0)
ax_2 = axes[2]
im_2 = ax_2.imshow(ours_mat, cmap="turbo", norm=color_norm)
ax_2.set_xlabel("(c) Ours", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_2, ax=ax_2, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()


# raw 2, col 1
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
ax_3 = axes[3]
im_3 = ax_3.imshow(dlpu_wrapped_mat, cmap="twilight", norm=color_norm)
ax_3.set_xlabel("(d) Wrapped", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_3, ax=ax_3, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

# raw 2, col 2
color_norm = colors.Normalize(vmin=0)
ax_4 = axes[4]
im_4 = ax_4.imshow(dlpu_gt_mat, cmap="turbo", norm=color_norm)
ax_4.set_xlabel("(e) Gt", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_4, ax=ax_4, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(2*np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()


# raw 2, col 3
color_norm = colors.Normalize(vmin=0)
ax_5 = axes[5]
im_5 = ax_5.imshow(dlpu_ours_mat, cmap="turbo", norm=color_norm)
ax_5.set_xlabel("(f) Ours", fontsize=7, labelpad=6)
cbar = fig.colorbar(im_5, ax=ax_5, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(2*np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()