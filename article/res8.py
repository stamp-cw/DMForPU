#各种方法噪声SNR对比图

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from load_article_data import load_article_ours_snr_data

# d_dict = load_article_ours_snr_data()
# snr_0db = d_dict['0db']
# snr_5db = d_dict['5db']
# snr_10db = d_dict['10db']
# snr_20db = d_dict['20db']
# snr_30db = d_dict['30db']

# snr_level = ["0dB", "5dB", "10dB", "20dB", "30dB"]
snr_level = ["0", "5", "10", "20", "30"]
method = ["GT", "DLPU", "PUNet", "Restormer", "SNAPHU", "SQD-LSTM", "U3Net", "Uformer", "Ours"]

df = pd.DataFrame()
df['snr'] = pd.Series(snr_level).repeat(len(method))
# np.concatenate([a, b, d])
df['rmse'] = np.concatenate([
    [],#gt
    [],#dlpu
    [],#punet
    [],#restormer
    [],#snaphu
    [],#sqd_lstm
    [],#u3net
    [],#uformer
    [],#ours
])

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
    x="snr",
    # y="col_line",
    y="rmse",
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
