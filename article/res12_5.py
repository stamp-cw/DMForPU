import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from load_article_data import _to_numpy_2d

# ======================
# 1. 数据（换成你的 pred / gt）
# ======================
# np.random.seed(0)
# pred = np.random.randn(2000)
# gt = np.random.randn(2000)

dlpu_pred_batch_path = "data/dlpu/samples_0_1.pt"
dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']
dlpu_pred = _to_numpy_2d(dlpu_pred)

gt_mat_path = r"data/gt/gt.mat"
gt_mat = sio.loadmat(gt_mat_path)['gt']

# 残差（推荐用绝对值）
# error = np.abs(pred - gt)
error = np.abs(dlpu_pred - gt_mat)

error=error.flatten()


# low, high = np.percentile(error, [1, 99])
# # error_clip = error[(error >= low) & (error <= high)]
# error = error[(error >= low) & (error <= high)]


error = np.clip(error, 0, 2)

# ======================
# 2. Seaborn风格
# ======================
sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))

eps=0.005

# ======================
# 3. 直方图（百分比）
# ======================
ax = sns.histplot(
    error,
    bins=10,
    stat="percent",  # 关键：显示百分比
    edgecolor="blue",
    color="white",
    linewidth=1.5
)

plt.xlabel("Error / rad")
plt.ylabel("Precent (%)")


# ======================
# 4. 区间比例计算函数
# ======================
def calc_ratio(data, low=None, high=None):
    if low is None:
        return np.mean(data <= high)
    if high is None:
        return np.mean(data >= low)
    return np.mean((data >= low) & (data <= high))


# ======================
# 5. 括号标注函数
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
            ha='center', color='red', fontsize=10)


# ======================
# 6. 定义区间（你可以改这里）
# ======================
ranges = [
    (0, 0.2),
    (1, 2),
    # (1, None)
]

# ======================
# 7. 添加标注
# ======================
y_max = ax.get_ylim()[1]
y_base = y_max * 0.6
y_step = y_max * 0.15

for i, (low, high) in enumerate(ranges):
    ratio = calc_ratio(error, low, high) * 100
    # y = y_base + i * y_step
    # y = y_max
    # y = y_max
    y = ratio + y_max * 0.05

    label = f"{ratio:.1f}%"

    add_range_annotation(ax, low, high, y, label)

# ======================
# 8. 美化
# ======================
plt.xlim(0-eps, 2+eps)
# plt.xlim(0, 5)
# plt.ylim(0, y_max * 1.2)
plt.ylim(0, y_max * 1.2)

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# ======================
# 9. 显示
# ======================
plt.show()