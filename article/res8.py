# 选三张图，不同方法局部解缠放大图，并标出局部区域，展示不同方法在局部区域的表现差异
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
from spectral.io import envi
import matplotlib.patches as patches

from load_article_data import pi_formatter,_to_numpy_2d, load_article_data_idx, load_article_data_dlpu_idx

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

# 第一行
punet_pred_1 = load_article_data_idx("punet", 1)
u3net_pred_1 = load_article_data_idx("u3net", 1)
dlpu_pred_1 = load_article_data_idx("dlpu", 1)
sqd_lstm_pred_1 = load_article_data_idx("sqd_lstm", 1)
restormer_pred_1 = load_article_data_idx("restormer", 1)
uformer_pred_1 = load_article_data_idx("uformer", 1)
ours_pred_1 = load_article_data_idx("ours", 1)
wrapped_mat_1 = load_article_data_idx("wrapped", 1)
gt_mat_1 = load_article_data_idx("gt", 1)

# 第二行
punet_pred_2 = load_article_data_idx("punet", 2)
u3net_pred_2 = load_article_data_idx("u3net", 2)
dlpu_pred_2 = load_article_data_idx("dlpu", 2)
sqd_lstm_pred_2 = load_article_data_idx("sqd_lstm", 2)
restormer_pred_2 = load_article_data_idx("restormer", 2)
uformer_pred_2 = load_article_data_idx("uformer", 2)
ours_pred_2 = load_article_data_idx("ours", 2)
wrapped_mat_2 = load_article_data_idx("wrapped", 2)
gt_mat_2 = load_article_data_idx("gt", 2)


# 第三行
punet_pred_3 = load_article_data_idx("punet", 3)
u3net_pred_3 = load_article_data_idx("u3net", 3)
dlpu_pred_3 = load_article_data_idx("dlpu", 3)
sqd_lstm_pred_3 = load_article_data_idx("sqd_lstm", 3)
restormer_pred_3 = load_article_data_idx("restormer", 3)
uformer_pred_3 = load_article_data_idx("uformer", 3)
ours_pred_3 = load_article_data_idx("ours", 3)
wrapped_mat_3 = load_article_data_idx("wrapped", 3)
gt_mat_3 = load_article_data_idx("gt", 3)



# 第四行
punet_pred_dlpu_1 = load_article_data_dlpu_idx("punet", 1)
u3net_pred_dlpu_1 = load_article_data_dlpu_idx("u3net", 1)
dlpu_pred_dlpu_1 = load_article_data_dlpu_idx("dlpu", 1)
sqd_lstm_pred_dlpu_1 = load_article_data_dlpu_idx("sqd_lstm", 1)
restormer_pred_dlpu_1 = load_article_data_dlpu_idx("restormer", 1)
uformer_pred_dlpu_1 = load_article_data_dlpu_idx("uformer", 1)
ours_pred_dlpu_1 = load_article_data_dlpu_idx("ours", 1)
wrapped_mat_dlpu_1 = load_article_data_dlpu_idx("wrapped", 1)
gt_mat_dlpu_1 = load_article_data_dlpu_idx("gt", 1)


# 第五行
punet_pred_dlpu_2 = load_article_data_dlpu_idx("punet", 2)
u3net_pred_dlpu_2 = load_article_data_dlpu_idx("u3net", 2)
dlpu_pred_dlpu_2 = load_article_data_dlpu_idx("dlpu", 2)
sqd_lstm_pred_dlpu_2 = load_article_data_dlpu_idx("sqd_lstm", 2)
restormer_pred_dlpu_2 = load_article_data_dlpu_idx("restormer", 2)
uformer_pred_dlpu_2 = load_article_data_dlpu_idx("uformer", 2)
ours_pred_dlpu_2 = load_article_data_dlpu_idx("ours", 2)
wrapped_mat_dlpu_2 = load_article_data_dlpu_idx("wrapped", 2)
gt_mat_dlpu_2 = load_article_data_dlpu_idx("gt", 2)


# 第六行
punet_pred_dlpu_3 = load_article_data_dlpu_idx("punet", 3)
u3net_pred_dlpu_3 = load_article_data_dlpu_idx("u3net", 3)
dlpu_pred_dlpu_3 = load_article_data_dlpu_idx("dlpu", 3)
sqd_lstm_pred_dlpu_3 = load_article_data_dlpu_idx("sqd_lstm", 3)
restormer_pred_dlpu_3 = load_article_data_dlpu_idx("restormer", 3)
uformer_pred_dlpu_3 = load_article_data_dlpu_idx("uformer", 3)
ours_pred_dlpu_3 = load_article_data_dlpu_idx("ours", 3)
wrapped_mat_dlpu_3 = load_article_data_dlpu_idx("wrapped", 3)
gt_mat_dlpu_3 = load_article_data_dlpu_idx("gt", 3)


#
# # wrapped_mat_path1 = r"data/wrapped/wrapped.mat"
# wrapped_mat_path1 = r"data/wrapped/000001.mat"
# wrapped_mat1 = sio.loadmat(wrapped_mat_path1)['input']
#
# # gt_mat_path1 = r"data/gt/gt.mat"
# gt_mat_path1 = r"data/gt/000001.mat"
# gt_mat1 = sio.loadmat(gt_mat_path1)['gt']
#
#
# # keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
# snaphu_pred_path = "data/snaphu/000001.hdr"
# snaphu_mat = envi.open(snaphu_pred_path)
# snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)
#
# punet_pred_batch_path = "data/punet/samples_0_1.pt"
# punet_pred_batch_pt = torch.load(punet_pred_batch_path)
# punet_pred = punet_pred_batch_pt['pred_unwrapped']
#
# u3net_pred_batch_path = "data/u3net/samples_0_1.pt"
# u3net_pred_batch_pt = torch.load(u3net_pred_batch_path)
# u3net_pred = u3net_pred_batch_pt['pred_unwrapped']
#
# dlpu_pred_batch_path = "data/dlpu/samples_0_1.pt"
# dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
# dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']
#
# sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
# sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
# sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']
#
# restormer_pred_batch_path = "data/restormer/samples_0_1.pt"
# restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
# restormer_pred = restormer_pred_batch_pt['pred_unwrapped']
#
# uformer_pred_batch_path = "data/uformer/samples_0_1.pt"
# uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
# uformer_pred = uformer_pred_batch_pt['pred_unwrapped']
#
# ours_pred_batch_path = "data/ours/samples_0_1.pt"
# ours_pred_batch_pt = torch.load(ours_pred_batch_path)
# ours_pred = ours_pred_batch_pt['pred_unwrapped']

titles = [
    # 第一行
    "(a) Wrapped",
    "(b) GT",
    # "(c) SNAPHU",
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
    wrapped_mat_1,
    gt_mat_1,
    _to_numpy_2d(punet_pred_1),
    _to_numpy_2d(u3net_pred_1),
    _to_numpy_2d(dlpu_pred_1),
    _to_numpy_2d(sqd_lstm_pred_1),
    _to_numpy_2d(restormer_pred_1),
    _to_numpy_2d(uformer_pred_1),
    _to_numpy_2d(ours_pred_1),
    # 第二行
    wrapped_mat_2,
    gt_mat_2,
    _to_numpy_2d(punet_pred_2),
    _to_numpy_2d(u3net_pred_2),
    _to_numpy_2d(dlpu_pred_2),
    _to_numpy_2d(sqd_lstm_pred_2),
    _to_numpy_2d(restormer_pred_2),
    _to_numpy_2d(uformer_pred_2),
    _to_numpy_2d(ours_pred_2),
    # 第三行
    wrapped_mat_3,
    gt_mat_3,
    _to_numpy_2d(punet_pred_3),
    _to_numpy_2d(u3net_pred_3),
    _to_numpy_2d(dlpu_pred_3),
    _to_numpy_2d(sqd_lstm_pred_3),
    _to_numpy_2d(restormer_pred_3),
    _to_numpy_2d(uformer_pred_3),
    _to_numpy_2d(ours_pred_3),
    # 第四行
    wrapped_mat_dlpu_1,
    gt_mat_dlpu_1,
    _to_numpy_2d(punet_pred_dlpu_1),
    _to_numpy_2d(u3net_pred_dlpu_1),
    _to_numpy_2d(dlpu_pred_dlpu_1),
    _to_numpy_2d(sqd_lstm_pred_dlpu_1),
    _to_numpy_2d(restormer_pred_dlpu_1),
    _to_numpy_2d(uformer_pred_dlpu_1),
    _to_numpy_2d(ours_pred_dlpu_1),
    # 第五行
    wrapped_mat_dlpu_2,
    gt_mat_dlpu_2,
    _to_numpy_2d(punet_pred_dlpu_2),
    _to_numpy_2d(u3net_pred_dlpu_2),
    _to_numpy_2d(dlpu_pred_dlpu_2),
    _to_numpy_2d(sqd_lstm_pred_dlpu_2),
    _to_numpy_2d(restormer_pred_dlpu_2),
    _to_numpy_2d(uformer_pred_dlpu_2),
    _to_numpy_2d(ours_pred_dlpu_2),
    # 第六行
    wrapped_mat_dlpu_3,
    gt_mat_dlpu_3,
    _to_numpy_2d(punet_pred_dlpu_3),
    _to_numpy_2d(u3net_pred_dlpu_3),
    _to_numpy_2d(dlpu_pred_dlpu_3),
    _to_numpy_2d(sqd_lstm_pred_dlpu_3),
    _to_numpy_2d(restormer_pred_dlpu_3),
    _to_numpy_2d(uformer_pred_dlpu_3),
    _to_numpy_2d(ours_pred_dlpu_3),
]
cmaps = [
    # 第一行
    "twilight","turbo",
    "turbo","turbo","turbo","turbo",
    "turbo","turbo","turbo"
] * 6

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res8/figure.pdf"
png_img_path = r"res/res8/figure.png"
raw = 6
col = 9
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))

# # 第一行 与 第四行
# color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
# for ax, img, title, cmap in  zip_list[0:1] + zip_list[3*col:3*col+1]:
#     im = ax.imshow(img, cmap=cmap, norm=color_norm)
#     ax.set_title(title)
#     ax.set_axis_off()
#
# color_norm = colors.Normalize(vmin=0)
# for ax, img, title, cmap in zip_list[1:col] + zip_list[3*col+1:4*col]:
#     im = ax.imshow(img, cmap=cmap, norm=color_norm)
#     rect = patches.Rectangle(
#         (0, 0), 32, 32,
#         linewidth=3,
#         edgecolor='red',
#         facecolor='none'
#     )
#     ax.add_patch(rect)
#     ax.set_title(title)
#     ax.set_axis_off()

# 前三行
# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[0:1] + zip_list[col:col+1] + zip_list[2*col:2*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_axis_off()
    # ax.set_ylabel(title, fontsize=7, labelpad=6)

# Phase
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[1:col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        (0, 0), 32, 32,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[col+1:2*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        (0, 0), 32, 32,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[2*col+1:3*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        (0, 0), 32, 32,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

# 后三行
# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[3*col:3*col+1] + zip_list[4*col:4*col+1] + zip_list[5*col:5*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_axis_off()
    # ax.set_ylabel(title, fontsize=7, labelpad=6)

# Phase
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[3*col+1:4*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        # (0, 0), 32, 32,
        (0, 0), 64, 64,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[4*col+1:5*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        # (0, 0), 32, 32,
        (0, 0), 64, 64,
        linewidth=3,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_axis_off()

color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[5*col+1:6*col]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    rect = patches.Rectangle(
        # (0, 0), 32, 32,
        (0, 0), 64, 64,
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