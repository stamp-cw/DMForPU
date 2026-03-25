# 对比图，第一行：GT, SNAPHU, PUNet, U3Net, DLPU, SQD-LSTM, Restormer, Uformer, Ours
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
from spectral.io import envi
from article.load_article_data import _to_numpy_2d, pi_formatter

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

# synpu
wrapped_mat_path = r"data/wrapped/wrapped.mat"
gt_mat_path = r"data/gt/gt.mat"
wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
gt_mat = sio.loadmat(gt_mat_path)['gt']

# keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
# snaphu_pred_path = "data/snaphu/000001.hdr"
# snaphu_mat = envi.open(snaphu_pred_path)
# snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)

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


# insar_dlpu
dlpu_wrapped_mat_path = r"data_dlpu/wrapped/wrapped.mat"
dlpu_gt_mat_path = r"data_dlpu/gt/gt.mat"
dlpu_wrapped_mat = sio.loadmat(dlpu_wrapped_mat_path)['input']
# gt_mat = sio.loadmat(gt_mat_path)['gt']
dlpu_gt_mat = sio.loadmat(dlpu_gt_mat_path)['output']

# keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
dlpu_snaphu_pred_path = "data_dlpu/snaphu/000001.hdr"
dlpu_snaphu_mat = envi.open(dlpu_snaphu_pred_path)
dlpu_snaphu_pred = dlpu_snaphu_mat.load()[:,:,0].squeeze(-1)
# dlpu_snaphu_pred = dlpu_gt_mat

dlpu_punet_pred_batch_path = "data_dlpu/punet/samples_0_1.pt"
dlpu_punet_pred_batch_pt = torch.load(dlpu_punet_pred_batch_path)
dlpu_punet_pred = dlpu_punet_pred_batch_pt['pred_unwrapped']

dlpu_u3net_pred_batch_path = "data_dlpu/u3net/samples_0_1.pt"
dlpu_u3net_pred_batch_pt = torch.load(dlpu_u3net_pred_batch_path)
dlpu_u3net_pred = dlpu_u3net_pred_batch_pt['pred_unwrapped']

dlpu_dlpu_pred_batch_path = "data_dlpu/dlpu/samples_0_1.pt"
dlpu_dlpu_pred_batch_pt = torch.load(dlpu_dlpu_pred_batch_path)
dlpu_dlpu_pred = dlpu_dlpu_pred_batch_pt['pred_unwrapped']

dlpu_sqd_lstm_pred_batch_path = "data_dlpu/sqd_lstm/samples_0_1.pt"
dlpu_sqd_lstm_pred_batch_pt = torch.load(dlpu_sqd_lstm_pred_batch_path)
dlpu_sqd_lstm_pred = dlpu_sqd_lstm_pred_batch_pt['pred_unwrapped']

dlpu_restormer_pred_batch_path = "data_dlpu/restormer/samples_0_1.pt"
dlpu_restormer_pred_batch_pt = torch.load(dlpu_restormer_pred_batch_path)
dlpu_restormer_pred = dlpu_restormer_pred_batch_pt['pred_unwrapped']

dlpu_uformer_pred_batch_path = "data_dlpu/uformer/samples_0_1.pt"
dlpu_uformer_pred_batch_pt = torch.load(dlpu_uformer_pred_batch_path)
dlpu_uformer_pred = dlpu_uformer_pred_batch_pt['pred_unwrapped']

dlpu_ours_pred_batch_path = "data_dlpu/ours/samples_0_1.pt"
dlpu_ours_pred_batch_pt = torch.load(dlpu_ours_pred_batch_path)
dlpu_ours_pred = dlpu_ours_pred_batch_pt['pred_unwrapped']

titles = [
    # 第一行
    "(Sa) GT",
    # "(Sb) SNAPHU",
    "(Sb) PUNet",
    "(Sc) U3Net",
    "(Sd) DLPU",
    "(Se) SQD-LSTM",
    "(Sf) Restormer",
    "(Sg) Uformer",
    "(Sh) Ours",
    # 第二行
    "(Si) Wrapped",
    # "(Sk) Error SNAPHU",
    "(Sj) Error PUNet",
    "(Sk) Error U3Net",
    "(Sl) Error DLPU",
    "(Sm) Error SQD-LSTM",
    "(Sn) Error Restormer",
    "(So) Error Uformer",
    "(Sp) Error Ours",
    # 第三行
    "(Ia) GT",
    # "(Ib) SNAPHU",
    "(Ib) PUNet",
    "(Ic) U3Net",
    "(Id) DLPU",
    "(Ie) SQD-LSTM",
    "(If) Restormer",
    "(Ig) Uformer",
    "(Ih) Ours",
    # 第四行
    "(Ii) Wrapped",
    # "(Ik) Error SNAPHU",
    "(Ij) Error PUNet",
    "(Ik) Error U3Net",
    "(Il) Error DLPU",
    "(Im) Error SQD-LSTM",
    "(In) Error Restormer",
    "(Io) Error Uformer",
    "(Ip) Error Ours"
]
imgs = [
    # 第一行
    gt_mat,
    # snaphu_pred,
    _to_numpy_2d(punet_pred),
    _to_numpy_2d(u3net_pred),
    _to_numpy_2d(dlpu_pred),
    _to_numpy_2d(sqd_lstm_pred),
    _to_numpy_2d(restormer_pred),
    _to_numpy_2d(uformer_pred),
    _to_numpy_2d(ours_pred),
    # 第二行
    wrapped_mat,
    # gt_mat - snaphu_pred,
    gt_mat - _to_numpy_2d(punet_pred),
    gt_mat - _to_numpy_2d(u3net_pred),
    gt_mat - _to_numpy_2d(dlpu_pred),
    gt_mat - _to_numpy_2d(sqd_lstm_pred),
    gt_mat - _to_numpy_2d(restormer_pred),
    gt_mat - _to_numpy_2d(uformer_pred),
    gt_mat - _to_numpy_2d(ours_pred),

    # 第三行
    dlpu_gt_mat,
    # dlpu_snaphu_pred,
    _to_numpy_2d(dlpu_punet_pred),
    _to_numpy_2d(dlpu_u3net_pred),
    _to_numpy_2d(dlpu_dlpu_pred),
    _to_numpy_2d(dlpu_sqd_lstm_pred),
    _to_numpy_2d(dlpu_restormer_pred),
    _to_numpy_2d(dlpu_uformer_pred),
    _to_numpy_2d(dlpu_ours_pred),
    # 第四行
    dlpu_wrapped_mat,
    # dlpu_gt_mat - dlpu_snaphu_pred,
    dlpu_gt_mat - _to_numpy_2d(dlpu_punet_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_u3net_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_dlpu_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_sqd_lstm_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_restormer_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_uformer_pred),
    dlpu_gt_mat - _to_numpy_2d(dlpu_ours_pred)
]
cmaps = [
    # 第一行
    "turbo",
    "turbo","turbo","turbo","turbo",
    "turbo","turbo","turbo",
    # 第二行
    "twilight",
    "inferno","inferno","inferno","inferno",
    "inferno","inferno","inferno",
    # 第三行
    "turbo",
    "turbo","turbo","turbo","turbo",
    "turbo","turbo","turbo",
    # 第四行
    "twilight",
    "inferno","inferno","inferno","inferno",
    "inferno","inferno","inferno",
]

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res2/figure.pdf"
png_img_path = r"res/res2/figure.png"
raw = 2 * 2
col = 8
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))
# Synthetic [0:18] InSar-DLPU [18:]
# Phase
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[:col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[2*col:3*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(2*np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()

# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[col:col+1] + zip_list[3*col:3*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()
# Error
for ax, img, title, cmap in zip_list[col+1:2*col]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax, img, title, cmap in zip_list[3*col+1:]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()