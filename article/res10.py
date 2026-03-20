# 不同噪声下各个方法的局部解缠放大图
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch
from spectral.io import envi
import matplotlib.patches as patches

from article.load_article_data import _to_numpy_2d, pi_formatter, load_article_snr_data
from article.res2 import snaphu_pred

wrapped_mat_path = r"data/wrapped/wrapped.mat"
gt_mat_path = r"data/gt/gt.mat"
# wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
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
# 0db
punet_pred_0db = load_article_snr_data("punet", 0)
u3net_pred_0db = load_article_snr_data("u3net", 0)
dlpu_pred_0db = load_article_snr_data("dlpu", 0)
sqd_lstm_pred_0db = load_article_snr_data("sqd_lstm", 0)
restormer_pred_0db = load_article_snr_data("restormer", 0)
uformer_pred_0db = load_article_snr_data("uformer", 0)
ours_pred_0db = load_article_snr_data("ours", 0)
snaphu_pred_0db = load_article_snr_data("snaphu", 0)
wrapped_mat_0db = load_article_snr_data("wrapped", 0)

# 5db
punet_pred_5db = load_article_snr_data("punet", 5)
u3net_pred_5db = load_article_snr_data("u3net", 5)
dlpu_pred_5db = load_article_snr_data("dlpu", 5)
sqd_lstm_pred_5db = load_article_snr_data("sqd_lstm", 5)
restormer_pred_5db = load_article_snr_data("restormer", 5)
uformer_pred_5db = load_article_snr_data("uformer", 5)
ours_pred_5db = load_article_snr_data("ours", 5)
snaphu_pred_5db = load_article_snr_data("snaphu", 5)
wrapped_mat_5db = load_article_snr_data("wrapped", 5)

# 10db
punet_pred_10db = load_article_snr_data("punet", 10)
u3net_pred_10db = load_article_snr_data("u3net", 10)
dlpu_pred_10db = load_article_snr_data("dlpu", 10)
sqd_lstm_pred_10db = load_article_snr_data("sqd_lstm", 10)
restormer_pred_10db = load_article_snr_data("restormer", 10)
uformer_pred_10db = load_article_snr_data("uformer", 10)
ours_pred_10db = load_article_snr_data("ours", 10)
snaphu_pred_10db = load_article_snr_data("snaphu", 10)
wrapped_mat_10db = load_article_snr_data("wrapped", 10)

# 20db
punet_pred_20db = load_article_snr_data("punet", 20)
u3net_pred_20db = load_article_snr_data("u3net", 20)
dlpu_pred_20db = load_article_snr_data("dlpu", 20)
sqd_lstm_pred_20db = load_article_snr_data("sqd_lstm", 20)
restormer_pred_20db = load_article_snr_data("restormer", 20)
uformer_pred_20db = load_article_snr_data("uformer", 20)
ours_pred_20db = load_article_snr_data("ours", 20)
snaphu_pred_20db = load_article_snr_data("snaphu", 20)
wrapped_mat_20db = load_article_snr_data("wrapped", 20)

# 30db
punet_pred_30db = load_article_snr_data("punet", 30)
u3net_pred_30db = load_article_snr_data("u3net", 30)
dlpu_pred_30db = load_article_snr_data("dlpu", 30)
sqd_lstm_pred_30db = load_article_snr_data("sqd_lstm", 30)
restormer_pred_30db = load_article_snr_data("restormer", 30)
uformer_pred_30db = load_article_snr_data("uformer", 30)
ours_pred_30db = load_article_snr_data("ours", 30)
snaphu_pred_30db = load_article_snr_data("snaphu", 30)
wrapped_mat_30db = load_article_snr_data("wrapped", 30)

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
         ] * 5
imgs = [
           # 第一行 0db
           wrapped_mat_0db,
           gt_mat,
           snaphu_pred_0db,
           _to_numpy_2d(punet_pred_0db),
           _to_numpy_2d(u3net_pred_0db),
           _to_numpy_2d(dlpu_pred_0db),
           _to_numpy_2d(sqd_lstm_pred_0db),
           _to_numpy_2d(restormer_pred_0db),
           _to_numpy_2d(uformer_pred_0db),
           _to_numpy_2d(ours_pred_0db),
            # 第二行 5db
            wrapped_mat_5db,
            gt_mat,
            snaphu_pred_5db,
            _to_numpy_2d(punet_pred_5db),
            _to_numpy_2d(u3net_pred_5db),
            _to_numpy_2d(dlpu_pred_5db),
            _to_numpy_2d(sqd_lstm_pred_5db),
            _to_numpy_2d(restormer_pred_5db),
            _to_numpy_2d(uformer_pred_5db),
            _to_numpy_2d(ours_pred_5db),
            # 第三行 10db
            wrapped_mat_10db,
            gt_mat,
            snaphu_pred_10db,
            _to_numpy_2d(punet_pred_10db),
            _to_numpy_2d(u3net_pred_10db),
            _to_numpy_2d(dlpu_pred_10db),
            _to_numpy_2d(sqd_lstm_pred_10db),
            _to_numpy_2d(restormer_pred_10db),
            _to_numpy_2d(uformer_pred_10db),
            _to_numpy_2d(ours_pred_10db),
            # 第四行 20db
            wrapped_mat_20db,
            gt_mat,
            snaphu_pred_20db,
            _to_numpy_2d(punet_pred_20db),
            _to_numpy_2d(u3net_pred_20db),
            _to_numpy_2d(dlpu_pred_20db),
            _to_numpy_2d(sqd_lstm_pred_20db),
            _to_numpy_2d(restormer_pred_20db),
            _to_numpy_2d(uformer_pred_20db),
            _to_numpy_2d(ours_pred_20db),
            # 第五行 30db
            wrapped_mat_30db,
            gt_mat,
            snaphu_pred_30db,
            _to_numpy_2d(punet_pred_30db),
            _to_numpy_2d(u3net_pred_30db),
            _to_numpy_2d(dlpu_pred_30db),
            _to_numpy_2d(sqd_lstm_pred_30db),
            _to_numpy_2d(restormer_pred_30db),
            _to_numpy_2d(uformer_pred_30db),
            _to_numpy_2d(ours_pred_30db),
]
cmaps = [
            # 第一行
            "twilight","turbo",
            "turbo","turbo","turbo","turbo",
            "turbo","turbo","turbo","turbo",
        ] * 5

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
pdf_img_path = r"res/res10/figure.pdf"
png_img_path = r"res/res10/figure.png"
raw = 5
col = 10
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()
zip_list = list(zip(axes, imgs, titles, cmaps))
# Synthetic [0:18] InSar-DLPU [18:]

# # 第一行
# color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
# for ax, img, title, cmap in  zip_list[0:1]:
#     im = ax.imshow(img, cmap=cmap, norm=color_norm)
#     ax.set_title(title)
#     ax.set_axis_off()
#
# color_norm = colors.Normalize(vmin=0)
# for ax, img, title, cmap in zip_list[1:col]:
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

# wrapped
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
for ax, img, title, cmap in  zip_list[0:1] + zip_list[col:col+1] + zip_list[2*col:2*col+1] + zip_list[3*col:3*col+1] + zip_list[4*col:4*col+1]:
    im = ax.imshow(img, cmap=cmap, norm=color_norm)
    ax.set_axis_off()
    # ax.set_ylabel(title, fontsize=7, labelpad=6)

# Phase
color_norm = colors.Normalize(vmin=0)
for ax, img, title, cmap in zip_list[1:col] + zip_list[col+1:2*col] + zip_list[2*col+1:3*col] + zip_list[3*col+1:4*col] + zip_list[4*col+1:5*col]:
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