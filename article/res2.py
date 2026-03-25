# 对比图，第一行：GT, SNAPHU, PUNet, U3Net, DLPU, SQD-LSTM, Restormer, Uformer, Ours
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
from spectral.io import envi
from article.load_article_data import _to_numpy_2d, pi_formatter
import seaborn as sns

# ======================
# 图设置
# ======================

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

# ======================
# 数据准备
# ======================



# synpu

clip_min = 0
clip_max = 2

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
punet_error = gt_mat - _to_numpy_2d(punet_pred)
punet_clip_error = np.clip(np.abs(punet_error).flatten(), clip_min, clip_max)

u3net_pred_batch_path = "data/u3net/samples_0_1.pt"
u3net_pred_batch_pt = torch.load(u3net_pred_batch_path)
u3net_pred = u3net_pred_batch_pt['pred_unwrapped']
u3net_error = gt_mat - _to_numpy_2d(u3net_pred)
u3net_clip_error = np.clip(np.abs(u3net_error).flatten(), clip_min, clip_max)

dlpu_pred_batch_path = "data/dlpu/samples_0_1.pt"
dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']
dlpu_error = gt_mat - _to_numpy_2d(dlpu_pred)
dlpu_clip_error = np.clip(np.abs(dlpu_error).flatten(), clip_min, clip_max)


sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']
sqd_lstm_error = gt_mat - _to_numpy_2d(sqd_lstm_pred)
sqd_lstm_clip_error = np.clip(np.abs(sqd_lstm_error).flatten(), clip_min, clip_max)


restormer_pred_batch_path = "data/restormer/samples_0_1.pt"
restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
restormer_pred = restormer_pred_batch_pt['pred_unwrapped']
restormer_error = gt_mat - _to_numpy_2d(restormer_pred)
restormer_clip_error = np.clip(np.abs(restormer_error).flatten(), clip_min, clip_max)


uformer_pred_batch_path = "data/uformer/samples_0_1.pt"
uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
uformer_pred = uformer_pred_batch_pt['pred_unwrapped']
uformer_error = gt_mat - _to_numpy_2d(uformer_pred)
uformer_clip_error = np.clip(np.abs(uformer_error).flatten(), clip_min, clip_max)


ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']
ours_error = gt_mat - _to_numpy_2d(ours_pred)
ours_clip_error = np.clip(np.abs(ours_error).flatten(), clip_min, clip_max)

# insar_dlpu

dlpu_clip_min = 0
dlpu_clip_max = 25
# dlpu_clip_max = 24

dlpu_wrapped_mat_path = r"data_dlpu/wrapped/wrapped.mat"
dlpu_gt_mat_path = r"data_dlpu/gt/gt.mat"
dlpu_wrapped_mat = sio.loadmat(dlpu_wrapped_mat_path)['input']
# gt_mat = sio.loadmat(gt_mat_path)['gt']
dlpu_gt_mat = sio.loadmat(dlpu_gt_mat_path)['output']

# # keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
# dlpu_snaphu_pred_path = "data_dlpu/snaphu/000001.hdr"
# dlpu_snaphu_mat = envi.open(dlpu_snaphu_pred_path)
# dlpu_snaphu_pred = dlpu_snaphu_mat.load()[:,:,0].squeeze(-1)
# # dlpu_snaphu_pred = dlpu_gt_mat

dlpu_punet_pred_batch_path = "data_dlpu/punet/samples_0_1.pt"
dlpu_punet_pred_batch_pt = torch.load(dlpu_punet_pred_batch_path)
dlpu_punet_pred = dlpu_punet_pred_batch_pt['pred_unwrapped']
dlpu_punet_error = dlpu_gt_mat - _to_numpy_2d(dlpu_punet_pred)
dlpu_punet_clip_error = np.clip(np.abs(dlpu_punet_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_u3net_pred_batch_path = "data_dlpu/u3net/samples_0_1.pt"
dlpu_u3net_pred_batch_pt = torch.load(dlpu_u3net_pred_batch_path)
dlpu_u3net_pred = dlpu_u3net_pred_batch_pt['pred_unwrapped']
dlpu_u3net_error =dlpu_gt_mat - _to_numpy_2d(dlpu_u3net_pred)
dlpu_u3net_clip_error = np.clip(np.abs(dlpu_u3net_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_dlpu_pred_batch_path = "data_dlpu/dlpu/samples_0_1.pt"
dlpu_dlpu_pred_batch_pt = torch.load(dlpu_dlpu_pred_batch_path)
dlpu_dlpu_pred = dlpu_dlpu_pred_batch_pt['pred_unwrapped']
dlpu_dlpu_error = dlpu_gt_mat - _to_numpy_2d(dlpu_dlpu_pred)
dlpu_dlpu_clip_error = np.clip(np.abs(dlpu_dlpu_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_sqd_lstm_pred_batch_path = "data_dlpu/sqd_lstm/samples_0_1.pt"
dlpu_sqd_lstm_pred_batch_pt = torch.load(dlpu_sqd_lstm_pred_batch_path)
dlpu_sqd_lstm_pred = dlpu_sqd_lstm_pred_batch_pt['pred_unwrapped']
dlpu_sqd_lstm_error = dlpu_gt_mat - _to_numpy_2d(dlpu_sqd_lstm_pred)
dlpu_sqd_lstm_clip_error = np.clip(np.abs(dlpu_sqd_lstm_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_restormer_pred_batch_path = "data_dlpu/restormer/samples_0_1.pt"
dlpu_restormer_pred_batch_pt = torch.load(dlpu_restormer_pred_batch_path)
dlpu_restormer_pred = dlpu_restormer_pred_batch_pt['pred_unwrapped']
dlpu_restormer_error = dlpu_gt_mat - _to_numpy_2d(dlpu_restormer_pred)
dlpu_restormer_clip_error = np.clip(np.abs(dlpu_restormer_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_uformer_pred_batch_path = "data_dlpu/uformer/samples_0_1.pt"
dlpu_uformer_pred_batch_pt = torch.load(dlpu_uformer_pred_batch_path)
dlpu_uformer_pred = dlpu_uformer_pred_batch_pt['pred_unwrapped']
dlpu_uformer_error = dlpu_gt_mat - _to_numpy_2d(dlpu_uformer_pred)
dlpu_uformer_clip_error = np.clip(np.abs(dlpu_uformer_error).flatten(), dlpu_clip_min, dlpu_clip_max)


dlpu_ours_pred_batch_path = "data_dlpu/ours/samples_0_1.pt"
dlpu_ours_pred_batch_pt = torch.load(dlpu_ours_pred_batch_path)
dlpu_ours_pred = dlpu_ours_pred_batch_pt['pred_unwrapped']
dlpu_ours_error = dlpu_gt_mat - _to_numpy_2d(dlpu_ours_pred)
dlpu_ours_clip_error = np.clip(np.abs(dlpu_ours_error).flatten(), dlpu_clip_min, dlpu_clip_max)

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
    "(Sq) Null",
    "(Sr) Error PUNet / rad",
    "(Ss) Error U3Net / rad",
    "(St) Error DLPU / rad",
    "(Su) Error SQD-LSTM / rad",
    "(Sv) Error Restormer / rad",
    "(Sw) Error Uformer / rad",
    "(Sx) Error Ours / rad",

    # 第四行
    "(Ia) GT",
    # "(Ib) SNAPHU",
    "(Ib) PUNet",
    "(Ic) U3Net",
    "(Id) DLPU",
    "(Ie) SQD-LSTM",
    "(If) Restormer",
    "(Ig) Uformer",
    "(Ih) Ours",

    # 第五行
    "(Ii) Wrapped",
    # "(Ik) Error SNAPHU",
    "(Ij) Error PUNet",
    "(Ik) Error U3Net",
    "(Il) Error DLPU",
    "(Im) Error SQD-LSTM",
    "(In) Error Restormer",
    "(Io) Error Uformer",
    "(Ip) Error Ours",

    # 第六行
    "(Iq) Null",
    "(Ir) Error PUNet / rad",
    "(Is) Error U3Net / rad",
    "(It) Error DLPU / rad",
    "(Iu) Error SQD-LSTM / rad",
    "(Iv) Error Restormer / rad",
    "(Iw) Error Uformer / rad",
    "(Ix) Error Ours / rad",
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
    punet_error,
    u3net_error,
    dlpu_error,
    sqd_lstm_error,
    restormer_error,
    uformer_error,
    ours_error,

    # 第三行
    wrapped_mat,
    punet_clip_error,
    u3net_clip_error,
    dlpu_clip_error,
    sqd_lstm_clip_error,
    restormer_clip_error,
    uformer_clip_error,
    ours_clip_error,

    # 第四行
    dlpu_gt_mat,
    # dlpu_snaphu_pred,
    _to_numpy_2d(dlpu_punet_pred),
    _to_numpy_2d(dlpu_u3net_pred),
    _to_numpy_2d(dlpu_dlpu_pred),
    _to_numpy_2d(dlpu_sqd_lstm_pred),
    _to_numpy_2d(dlpu_restormer_pred),
    _to_numpy_2d(dlpu_uformer_pred),
    _to_numpy_2d(dlpu_ours_pred),

    # 第五行
    dlpu_wrapped_mat,
    # dlpu_gt_mat - dlpu_snaphu_pred,
    dlpu_punet_error,
    dlpu_u3net_error,
    dlpu_dlpu_error,
    dlpu_sqd_lstm_error,
    dlpu_restormer_error,
    dlpu_uformer_error,
    dlpu_ours_error,

    # 第六行
    dlpu_wrapped_mat,
    dlpu_punet_clip_error,
    dlpu_u3net_clip_error,
    dlpu_dlpu_clip_error,
    dlpu_sqd_lstm_clip_error,
    dlpu_restormer_clip_error,
    dlpu_uformer_clip_error,
    dlpu_ours_clip_error,
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
    "turbo",
    "turbo", "turbo", "turbo", "turbo",
    "turbo", "turbo", "turbo",
    # 第五行
    "twilight",
    "inferno","inferno","inferno","inferno",
    "inferno","inferno","inferno",
    # 第六行
    "twilight",
    "inferno", "inferno", "inferno", "inferno",
    "inferno", "inferno", "inferno",
]

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res2/figure.pdf"
png_img_path = r"res/res2/figure.png"
# raw = 2 * 2
raw = 2 * 3
col = 8
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))

# ======================
# 2. Seaborn风格
# ======================
sns.set(style="whitegrid")
eps=0.005

# ======================
# 区间比例计算函数
# ======================
def calc_ratio(data, low=None, high=None):
    if low is None:
        return np.mean(data <= high)
    if high is None:
        return np.mean(data >= low)
    return np.mean((data >= low) & (data <= high))

# ======================
# 括号标注函数
# ======================
def add_range_annotation(ax, low, high, y, text):
    if high is None:
        high = ax.get_xlim()[1]

    # 横线
    # ax.hlines(y=y, xmin=low, xmax=high, colors='gray', linewidth=1.5)
    ax.hlines(y=y, xmin=low, xmax=high, colors='red', linewidth=1.5)

    # 两端“括号”
    # ax.vlines([low, high], y - 1, y, colors='gray', linewidth=1.5)
    ax.vlines([low, high], y - 1, y, colors='red', linewidth=1.5)

    # 百分比文字
    ax.text((low + high) / 2, y + 1, text,
            ha='center', color='red', fontsize=7)

# ======================
# 定义区间（你可以改这里）
# ======================
ranges = [
    # (0, 0.2),
    # (1, 2),
    (clip_min, clip_max / 10),
    (clip_max / 2, clip_max),
    # (1, None)
]

dlpu_ranges = [
    # (0, 2.5),
    # (12.5, 25),
    # (dlpu_clip_min, clip_max / 10),
    # (dlpu_clip_max / 2, clip_max),

    (dlpu_clip_min, dlpu_clip_max / 10),
    (dlpu_clip_max / 2, dlpu_clip_max),
    # (1, None)
]

# ======================
# 绘图
# ======================

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
# for ax, img, title, cmap in zip_list[2*col:3*col]:
for ax, img, title, cmap in zip_list[3*col:4*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(2*np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()

# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
# for ax, img, title, cmap in  zip_list[col:col+1] + zip_list[3*col:3*col+1]:
for ax, img, title, cmap in  zip_list[col:col+1] + zip_list[4*col:4*col+1]:
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

# for ax, img, title, cmap in zip_list[3*col+1:]:
for ax, img, title, cmap in zip_list[4*col+1:5*col]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Error histogram
for ax, error, title, cmap in zip_list[2*col+1:3*col]:
    sns.histplot(
        error,
        bins=10,
        stat="percent",  # 关键：显示百分比
        edgecolor="blue",
        color="white",
        linewidth=1.5,
        ax=ax
    )
    # ax.set_xlabel("Error / rad", fontsize=7, labelpad=6)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    ax.set_ylabel("Precent (%)", fontsize=7, labelpad=6)

    # ======================
    # 添加标注
    # ======================
    y_max = ax.get_ylim()[1]

    for i, (low, high) in enumerate(ranges):
        ratio = calc_ratio(error, low, high) * 100
        y = ratio + y_max * 0.05

        label = f"{ratio:.1f}%"

        add_range_annotation(ax, low, high, y, label)

    ax.set_xlim(clip_min-eps, clip_max+eps)
    ax.set_ylim(0, y_max * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

for ax, error, title, cmap in zip_list[5*col+1:]:
    sns.histplot(
        error,
        bins=10,
        stat="percent",  # 关键：显示百分比
        edgecolor="blue",
        color="white",
        linewidth=1.5,
        ax=ax
    )
    # ax.set_xlabel("Error / rad", fontsize=7, labelpad=6)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    ax.set_ylabel("Precent (%)", fontsize=7, labelpad=6)

    # ======================
    # 添加标注
    # ======================
    y_max = ax.get_ylim()[1]

    for i, (low, high) in enumerate(dlpu_ranges):
        ratio = calc_ratio(error, low, high) * 100
        y = ratio * 0.5

        label = f"{ratio:.1f}%"

        add_range_annotation(ax, low, high, y, label)

    ax.set_xlim(dlpu_clip_min-eps, dlpu_clip_max+eps)
    ax.set_ylim(0, y_max * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()