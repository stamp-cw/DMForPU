# 中心线逐像素相位对比图
# 中心线横轴逐像素相位对比
# 中心线纵轴逐像素相位对比
# 解缠
# 横轴：128,0:256
# 纵轴：0:256,128
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from load_article_data import load_article_data
from load_article_data import load_article_data_dlpu



palette = {
    "GT": "black",
    "Ours": "#d62728",   # 红色
    "DLPU": "#1f77b4",
    "PUNet": "#2ca02c",
    "Restormer": "#9467bd",
    "SNAPHU": "#8c564b",
    "SQD-LSTM": "#e377c2",
    "U3Net": "#7f7f7f",
    "Uformer": "#bcbd22",
}

fig_dpi = 600
fig_size_W = 3.5 * 2
fig_size_H = 2.5 * 2
pdf_img_path = r"res/res7/figure.pdf"
png_img_path = r"res/res7/figure.png"
raw = 1 * 2
col = 2

fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()

# Synthesis
d_dict = load_article_data()
gt=d_dict['gt']
dlpu_pred=d_dict['dlpu_pred']
punet_pred=d_dict['punet_pred']
restormer_pred=d_dict['restormer_pred']
# snaphu_pred=d_dict['snaphu_pred']
snaphu_pred=d_dict['gt']
sqd_lstm_pred=d_dict['sqd_lstm_pred']
u3net_pred=d_dict['u3net_pred']
uformer_pred=d_dict['uformer_pred']
ours_pred=d_dict['ours_pred']

method = ["GT", "DLPU", "PUNet", "Restormer", "SNAPHU", "SQD-LSTM", "U3Net", "Uformer", "Ours"]

raw_line_pos = 64
col_line_pos = 64
line_piex_num = 128

df = pd.DataFrame()
df['method'] = pd.Series(method).repeat(line_piex_num)
# np.concatenate([a, b, d])
df['col_line'] = np.concatenate([
    gt[col_line_pos, :],
    dlpu_pred[col_line_pos, :],
    punet_pred[col_line_pos, :],
    restormer_pred[col_line_pos, :],
    snaphu_pred[col_line_pos, :],
    sqd_lstm_pred[col_line_pos, :],
    u3net_pred[col_line_pos, :],
    uformer_pred[col_line_pos, :],
    ours_pred[col_line_pos, :]
])

df['raw_line'] = np.concatenate([
    gt[:, raw_line_pos],
    dlpu_pred[:, raw_line_pos],
    punet_pred[:, raw_line_pos],
    restormer_pred[:, raw_line_pos],
    snaphu_pred[:, raw_line_pos],
    sqd_lstm_pred[:, raw_line_pos],
    u3net_pred[:, raw_line_pos],
    uformer_pred[:, raw_line_pos],
    ours_pred[:, raw_line_pos]
])

df ['col_line_diff'] = df['col_line'] - np.tile(gt[col_line_pos, :], len(method))
df ['raw_line_diff'] = df['raw_line'] - np.tile(gt[:, raw_line_pos], len(method))
df['line_piex_idx'] = list(range(line_piex_num)) * len(method)

# raw line逐像素相位对比图
sns.lineplot(
    data=df,
    x="line_piex_idx",
    # y="col_line",
    # y="col_line_diff",
    y="raw_line_diff",
    hue="method",
    # style="method",
    markers=True,
    dashes=True,
    palette=palette,
    ax=axes[0]
)

# col line逐像素相位对比图
sns.lineplot(
    data=df,
    x="line_piex_idx",
    # y="col_line",
    y="col_line_diff",
    hue="method",
    # style="method",
    markers=True,
    dashes=True,
    palette=palette,
    ax=axes[1]
)

axes[0].legend(
    bbox_to_anchor=(1.02, 1),  # 向右移动
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

axes[1].legend(
    bbox_to_anchor=(1.02, 1),  # 向右移动
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

# InSar-DLPU
d_dict2 = load_article_data_dlpu()
gt=d_dict2['gt']
dlpu_pred=d_dict2['dlpu_pred']
punet_pred=d_dict2['punet_pred']
restormer_pred=d_dict2['restormer_pred']
# snaphu_pred=d_dict2['snaphu_pred']
snaphu_pred=d_dict2['gt']
sqd_lstm_pred=d_dict2['sqd_lstm_pred']
u3net_pred=d_dict2['u3net_pred']
uformer_pred=d_dict2['uformer_pred']
ours_pred=d_dict2['ours_pred']

method = ["GT", "DLPU", "PUNet", "Restormer", "SNAPHU", "SQD-LSTM", "U3Net", "Uformer", "Ours"]

raw_line_pos = 128
col_line_pos = 128
line_piex_num = 256

df = pd.DataFrame()
df['method'] = pd.Series(method).repeat(line_piex_num)
# np.concatenate([a, b, d])
df['col_line'] = np.concatenate([
    gt[col_line_pos, :],
    dlpu_pred[col_line_pos, :],
    punet_pred[col_line_pos, :],
    restormer_pred[col_line_pos, :],
    snaphu_pred[col_line_pos, :],
    sqd_lstm_pred[col_line_pos, :],
    u3net_pred[col_line_pos, :],
    uformer_pred[col_line_pos, :],
    ours_pred[col_line_pos, :]
])

df['raw_line'] = np.concatenate([
    gt[:, raw_line_pos],
    dlpu_pred[:, raw_line_pos],
    punet_pred[:, raw_line_pos],
    restormer_pred[:, raw_line_pos],
    snaphu_pred[:, raw_line_pos],
    sqd_lstm_pred[:, raw_line_pos],
    u3net_pred[:, raw_line_pos],
    uformer_pred[:, raw_line_pos],
    ours_pred[:, raw_line_pos]
])

df ['col_line_diff'] = df['col_line'] - np.tile(gt[col_line_pos, :], len(method))
df ['raw_line_diff'] = df['raw_line'] - np.tile(gt[:, raw_line_pos], len(method))
df['line_piex_idx'] = list(range(line_piex_num)) * len(method)

# raw line逐像素相位对比图
sns.lineplot(
    data=df,
    x="line_piex_idx",
    # y="col_line",
    # y="col_line_diff",
    y="raw_line_diff",
    hue="method",
    # style="method",
    markers=True,
    dashes=True,
    palette=palette,
    ax=axes[2]
)

# col line逐像素相位对比图
sns.lineplot(
    data=df,
    x="line_piex_idx",
    # y="col_line",
    y="col_line_diff",
    hue="method",
    # style="method",
    markers=True,
    dashes=True,
    palette=palette,
    ax=axes[3]
)

axes[2].legend(
    bbox_to_anchor=(1.02, 1),  # 向右移动
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

axes[3].legend(
    bbox_to_anchor=(1.02, 1),  # 向右移动
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()