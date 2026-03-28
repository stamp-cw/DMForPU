# 不同预测模式对比图，
# synpu: 第一列，wrapped ,第二列， pred， 第三列， error, 第四列， error histogram
# insardlpu: 第五列 wrapped ,第六列， pred， 第七列， error, 第八列， error histogram
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

ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']
ours_error = gt_mat - _to_numpy_2d(ours_pred)
ours_clip_error = np.clip(np.abs(ours_error).flatten(), clip_min, clip_max)

ours_v_pred_batch_path = "data/ours_v_pred/samples_0_1.pt"
ours_v_pred_batch_pt = torch.load(ours_v_pred_batch_path)
ours_v_pred = ours_v_pred_batch_pt['pred_unwrapped']
ours_v_error = gt_mat - _to_numpy_2d(ours_v_pred)
ours_v_clip_error = np.clip(np.abs(ours_v_error).flatten(), clip_min, clip_max)

ours_e_pred_batch_path = "data/ours_e_pred/samples_0_1.pt"
ours_e_pred_batch_pt = torch.load(ours_e_pred_batch_path)
ours_e_pred = ours_e_pred_batch_pt['pred_unwrapped']
ours_e_error = gt_mat - _to_numpy_2d(ours_e_pred)
ours_e_clip_error = np.clip(np.abs(ours_e_error).flatten(), clip_min, clip_max)


# insar_dlpu
dlpu_clip_min = 0
dlpu_clip_max = 25
# dlpu_clip_max = 24

dlpu_wrapped_mat_path = r"data_dlpu/wrapped/wrapped.mat"
dlpu_gt_mat_path = r"data_dlpu/gt/gt.mat"
dlpu_wrapped_mat = sio.loadmat(dlpu_wrapped_mat_path)['input']
# gt_mat = sio.loadmat(gt_mat_path)['gt']
dlpu_gt_mat = sio.loadmat(dlpu_gt_mat_path)['output']

dlpu_ours_pred_batch_path = "data_dlpu/ours/samples_0_1.pt"
dlpu_ours_pred_batch_pt = torch.load(dlpu_ours_pred_batch_path)
dlpu_ours_pred = dlpu_ours_pred_batch_pt['pred_unwrapped']
dlpu_ours_error = dlpu_gt_mat - _to_numpy_2d(dlpu_ours_pred)
dlpu_ours_clip_error = np.clip(np.abs(dlpu_ours_error).flatten(), dlpu_clip_min, dlpu_clip_max)

dlpu_ours_v_pred_batch_path = "data_dlpu/ours_v_pred/samples_0_1.pt"
dlpu_ours_v_pred_batch_pt = torch.load(dlpu_ours_v_pred_batch_path)
dlpu_ours_v_pred = dlpu_ours_v_pred_batch_pt['pred_unwrapped']
dlpu_ours_v_error = dlpu_gt_mat - _to_numpy_2d(dlpu_ours_v_pred)
dlpu_ours_v_clip_error = np.clip(np.abs(dlpu_ours_v_error).flatten(), dlpu_clip_min, dlpu_clip_max)

dlpu_ours_e_pred_batch_path = "data_dlpu/ours_e_pred/samples_0_1.pt"
dlpu_ours_e_pred_batch_pt = torch.load(dlpu_ours_e_pred_batch_path)
dlpu_ours_e_pred = dlpu_ours_e_pred_batch_pt['pred_unwrapped']
dlpu_ours_e_error = dlpu_gt_mat - _to_numpy_2d(dlpu_ours_e_pred)
dlpu_ours_e_clip_error = np.clip(np.abs(dlpu_ours_e_error).flatten(), dlpu_clip_min, dlpu_clip_max)


titles = [
    # 第一行
    "(Sa) GT",
    "(Sb) Ours Pred",
    "(Sc) Error",
    "(Sd) Error / rad",
    "(Se) GT",
    "(Sf) Ours Pred",
    "(Sg) Error",
    "(Sh) Error / rad",
    # 第二行
    "(Va) GT",
    "(Vb) Ours v-Pred",
    "(Vc) Error",
    "(Vd) Error / rad",
    "(Ve) GT",
    "(Vf) Ours v-Pred",
    "(Vg) Error",
    "(Vh) Error / rad",

    # 第三行
    "(Ea) GT",
    "(Eb) Ours e-Pred",
    "(Ec) Error",
    "(Ed) Error / rad",
    "(Ee) GT",
    "(Ef) Ours e-Pred",
    "(Eg) Error",
    "(Eh) Error / rad",
]
imgs = [
    # 第一行
    gt_mat,
    _to_numpy_2d(ours_pred),
    ours_error,
    ours_clip_error,
    dlpu_gt_mat,
    _to_numpy_2d(dlpu_ours_pred),
    dlpu_ours_error,
    dlpu_ours_clip_error,
    # 第二行
    gt_mat,
    _to_numpy_2d(ours_v_pred),
    ours_v_error,
    ours_v_clip_error,
    dlpu_gt_mat,
    _to_numpy_2d(dlpu_ours_v_pred),
    dlpu_ours_v_error,
    dlpu_ours_v_clip_error,
    # 第三行
    gt_mat,
    _to_numpy_2d(ours_e_pred),
    ours_e_error,
    ours_e_clip_error,
    dlpu_gt_mat,
    _to_numpy_2d(dlpu_ours_e_pred),
    dlpu_ours_e_error,
    dlpu_ours_e_clip_error,
]
cmaps = [
    # 第一行
    "turbo",
    "turbo","inferno","inferno",
    "turbo",
    "turbo","inferno","inferno",
] * 3

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res13/figure.pdf"
png_img_path = r"res/res13/figure.png"
raw = 3
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

# # GT
# color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
# for ax, img, title, cmap in zip_list[0:1] + zip_list[col:col+1] + zip_list[2*col:2*col+1] + zip_list[4:5] + zip_list[col+4:col+4+1] + zip_list[2*col+4:2*col+4+1]:
#     im = ax.imshow(img, cmap=cmap, norm=color_norm)
#     ax.set_xlabel(title, fontsize=7, labelpad=6)
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.locator = MultipleLocator(np.pi)
#     cbar.formatter = FuncFormatter(pi_formatter)
#     cbar.update_ticks()

# GT + Pred
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[0:1] + zip_list[col:col+1] + zip_list[2*col:2*col+1] + zip_list[1:2] + zip_list[col+1:col+2] + zip_list[2*col+1:2*col+2] :
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[4:5] + zip_list[col+4:col+4+1] + zip_list[2*col+4:2*col+4+1] + zip_list[5:6] + zip_list[col+5:col+6] + zip_list[2*col+5:2*col+6]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MultipleLocator(2*np.pi)
    cbar.formatter = FuncFormatter(pi_formatter)
    cbar.update_ticks()

# Error
for ax, img, title, cmap in zip_list[2:3] + zip_list[col+2:col+3] + zip_list[2*col+2:2*col+3]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# for ax, img, title, cmap in zip_list[3*col+1:]:
for ax, img, title, cmap in zip_list[6:7] + zip_list[col+6:col+7] + zip_list[2*col+6:2*col+7]:
    im = ax.imshow(img, cmap=cmap)
    ax.set_xlabel(title, fontsize=7, labelpad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Error histogram
for ax, error, title, cmap in zip_list[3:4] + zip_list[col+3:col+4] + zip_list[2*col+3:2*col+4]:
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

for ax, error, title, cmap in zip_list[7:8] + zip_list[col+7:col+8] + zip_list[2*col+7:2*col+8]:
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