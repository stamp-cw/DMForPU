# 不同方法局部解缠放大图
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
from spectral.io import envi
import matplotlib.patches as patches

def pi_formatter(x, pos):
    k = x / np.pi

    # 0
    if np.isclose(k, 0):
        return "0"

    # 接近整数倍 π
    if np.isclose(k, round(k)):
        k_int = int(round(k))
        if k_int == 1:
            return r"$\pi$"
        elif k_int == -1:
            return r"$-\pi$"
        else:
            return rf"${k_int}\pi$"

    # 非整数倍：显示数值（可改精度）
    return rf"${x:.2f}$"

wrapped_mat_path = r"data/wrapped/wrapped.mat"
gt_mat_path = r"data/gt/gt.mat"
wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
gt_mat = sio.loadmat(gt_mat_path)['gt']
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

# keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
snaphu_pred_path = "data/snaphu/000001.hdr"
snaphu_mat = envi.open(snaphu_pred_path)
snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)

punet_pred_batch_path = "data/punet/samples_0_1.pt"
punet_pred_batch_pt = torch.load(punet_pred_batch_path)
punet_pred = punet_pred_batch_pt['pred_unwrapped']

u3net_pred_batch_path = "data/u3net/samples_0_1.pt"
u3net_pred_batch_pt = torch.load(u3net_pred_batch_path)
u3net_pred = u3net_pred_batch_pt['pred_unwrapped']

dlpu_pred_batch_path = "data/dlpu/samples_0_1.pt"
dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']

sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']

restormer_pred_batch_path = "data/restormer/samples_0_1.pt"
restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
restormer_pred = restormer_pred_batch_pt['pred_unwrapped']

uformer_pred_batch_path = "data/uformer/samples_0_1.pt"
uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
uformer_pred = uformer_pred_batch_pt['pred_unwrapped']

ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']

def _to_numpy_2d(x: torch.Tensor):
    return x.detach().cpu().squeeze().numpy()

titles = [
    # 第一行
    "(a) Wrapped",
    "(b) GT",
    "(c) SNAPHU",
    "(d) PUNet",
    "(e) U3Net",
    "(f) DLPU",
    "(g) SQD-LSTM",
    "(h) Restormer",
    "(i) Uformer",
    "(j) Ours",
] * 6
imgs = [
    # 第一行
    wrapped_mat,
    gt_mat,
    snaphu_pred,
    _to_numpy_2d(punet_pred),
    _to_numpy_2d(u3net_pred),
    _to_numpy_2d(dlpu_pred),
    _to_numpy_2d(sqd_lstm_pred),
    _to_numpy_2d(restormer_pred),
    _to_numpy_2d(uformer_pred),
    _to_numpy_2d(ours_pred),
] * 6
cmaps = [
    # 第一行
    "twilight","turbo",
    "turbo","turbo","turbo","turbo",
    "turbo","turbo","turbo","turbo",
] * 6

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res8/figure.pdf"
png_img_path = r"res/res8/figure.png"
raw = 6
col = 10
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))
# Synthetic [0:18] InSar-DLPU [18:]

# 第一行 与 第四行
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[0:1] + zip_list[3*col:3*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_title(title)
    ax.set_axis_off()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[1:col] + zip_list[3*col+1:4*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        (0, 0), 32, 32,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(title)
    ax.set_axis_off()

# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[col:col+1] + zip_list[2*col:2*col+1] + zip_list[4*col:4*col+1] + zip_list[5*col:5*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_axis_off()

# Phase
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[col+1:2*col] + zip_list[2*col+1:3*col] + zip_list[4*col+1:5*col] + zip_list[5*col+1:6*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        (0, 0), 32, 32,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()