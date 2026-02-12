# 中心线逐像素相位对比图
# 中心线横轴逐像素相位对比
# 中心线纵轴逐像素相位对比
# 解缠
# 横轴：128,0:256
# 纵轴：0:256,128
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import torch
import scipy.io as sio
import numpy as np

from load_article_data import load_article_data

d_dict = load_article_data()
gt=d_dict['gt']
dlpu_pred=d_dict['dlpu_pred']
punet_pred=d_dict['punet_pred']
restormer_pred=d_dict['restormer_pred']
snaphu_pred=d_dict['snaphu_pred']
sqd_lstm_pred=d_dict['sqd_lstm_pred']
u3net_pred=d_dict['u3net_pred']
uformer_pred=d_dict['uformer_pred']
ours_pred=d_dict['ours_pred']

method = ["GT", "DLPU", "PUNet", "Restormer", "SNAPHU", "SQD-LSTM", "U3Net", "Uformer", "Ours"]


# raw_line_pos = 128
# # raw_line_piex_num = 256
# col_line_pos = 128
# # col_line_piex_num = 256
# line_piex_num = 256

raw_line_pos = 64
# raw_line_piex_num = 256
col_line_pos = 64
# col_line_piex_num = 256
line_piex_num = 128


# print(gt.shape)
# print(gt[col_line_pos, :].shape)

print(dlpu_pred.shape)
print(dlpu_pred[col_line_pos, :].shape)
print(pd.Series(dlpu_pred[col_line_pos, :]).T)


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

# df = pd.DataFrame({
#     "EncoderRMSE": [0.005, 0.01, 0.02,
#                     0.005, 0.01, 0.02,
#                     0.005, 0.01, 0.02],
#     "ExtendedRMSE": [0.85, 0.95, 1.15,      # RWA
#                      0.08, 0.10, 0.12,      # PCA
#                      0.09, 0.11, 0.13],     # RWA+PCA
#     # "Method": ["RWA"] * 3 + ["PCA"] * 3 + ["RWA+PCA"] * 3,
#     "Method": ["unwrapped_gt"] * 3 + ["unwrapped_pred"] * 3,
#     "CompactBands": ["3", "20", "20"] * 3
# })

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

sns.lineplot(
    data=df,
    x="line_piex_idx",
    # y="col_line",
    y="col_line_diff",
    hue="method",
    # style="method",
    markers=True,
    dashes=True,
    palette=palette
)

plt.legend(
    bbox_to_anchor=(1.02, 1),  # 向右移动
    loc="upper left",
    borderaxespad=0,
    frameon=False
)

plt.grid(True, linestyle="--", alpha=0.6)
plt.xlabel("Line Pixel Index")
plt.ylabel("Col Phase Error Value")
# plt.title("Extended RMSE Trend Lines ↓")
plt.tight_layout()
plt.show()
