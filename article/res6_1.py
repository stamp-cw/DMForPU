import matplotlib.pyplot as plt
import numpy as np

# 示例数据（根据你的图替换）
x = np.array([0.005, 0.01, 0.02])
y_rwa_pca = np.array([0.1, 0.12, 0.15])
y_rwa = np.array([0.8, 0.95, 1.2])
y_pca = np.array([0.08, 0.1, 0.12])

# 对应 marker
# marker_small = np.array([0, 1, 2])  # 假设前三个是 'Compact bands until 3'
marker_small = np.array([0])  # 假设前三个是 'Compact bands until 3'
# marker_large = np.array([0, 1, 2])  # 假设后三个是 'Compact bands until 20'
marker_large = np.array([1])  # 假设后三个是 'Compact bands until 20'

fig, ax = plt.subplots(figsize=(4.0, 3.5))

# 画折线
ax.plot(x, y_rwa_pca, 'r--', label='RWA+PCA')
ax.plot(x, y_rwa, 'b--', label='RWA')
ax.plot(x, y_pca, 'g--', label='PCA')

# 添加 marker
ax.scatter(x[marker_small], y_rwa_pca[marker_small], c='r', marker='o', s=50)
ax.scatter(x[marker_large], y_rwa_pca[marker_large], c='r', marker='s', s=50)

ax.scatter(x[marker_small], y_rwa[marker_small], c='b', marker='o', s=100)
ax.scatter(x[marker_large], y_rwa[marker_large], c='b', marker='s', s=200)

ax.scatter(x[marker_small], y_pca[marker_small], c='g', marker='o', s=50)
ax.scatter(x[marker_large], y_pca[marker_large], c='g', marker='s', s=50)

# 网格
ax.grid(True, linestyle='--', alpha=0.6)

# 图例（线 + marker 分开显示）
ax.legend(frameon=False)

# 坐标轴标签
ax.set_xlabel("Encoder-Decoder RMSE")
ax.set_ylabel("Extended RMSE")

# 主标题
ax.set_title("Extended RMSE Trend Lines ↓", fontsize=10)

# 子图标签
fig.text(0.5, -0.05, "(b)", ha='center', fontsize=10)

plt.tight_layout()
plt.show()
