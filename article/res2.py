import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
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


punet_pred_batch_path = "data/punet/samples_0_1.pt"
punet_pred_batch_pt = torch.load(punet_pred_batch_path)
punet_pred = punet_pred_batch_pt['pred_unwrapped']

sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
# keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
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
    "Wrapped", "GT", "PUNet",
    "SQD-LSTM", "Restormer",
    "Uformer", "Ours",
    "Wrapped", "GT",
    "Diff PUNet", "Diff SQD-LSTM",
    "Diff Restormer",
    "Diff Uformer", "Diff Ours"
]
imgs = [
    wrapped_mat, gt_mat, _to_numpy_2d(punet_pred), _to_numpy_2d(sqd_lstm_pred), _to_numpy_2d(restormer_pred),
    _to_numpy_2d(uformer_pred), _to_numpy_2d(ours_pred),
    wrapped_mat, gt_mat,
    gt_mat - _to_numpy_2d(punet_pred),
    gt_mat - _to_numpy_2d(sqd_lstm_pred),
    gt_mat - _to_numpy_2d(restormer_pred),
    gt_mat - _to_numpy_2d(uformer_pred),
    gt_mat - _to_numpy_2d(ours_pred)
]
cmaps = [
    "twilight", "turbo", "turbo",
    "turbo", "turbo",
    "turbo", "turbo",
    "twilight", "turbo",
    "inferno", "inferno", "inferno",
    "inferno", "inferno"
]

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
img_path = r"res/res2/figure.pdf"
raw = 2
col = 7
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))
# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[:1]+zip_list[col:col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()
color_norm = colors.Normalize(vmin=0)
ims = []
for ax, img, title, cmap in zip_list[1:col] + zip_list[col+1:col+2]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    ims.append(im)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()
for ax, img, title, cmap in zip_list[col+2:]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    ims.append(im)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()