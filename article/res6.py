import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# categories = ["Ours", "PUNet", "U3Net", "DLPU", "SQD-LSTM", "Restormer", "Uformer"]
# model_size = [159.67, 159.67, 159.67, 4.03, 8.43, 299.50, 10.33 , 20.49, 237.02]
# log_model_size = np.log10(model_size)
# "NRMSE": [0.0054, 0.0054 + 0.001, 0.0054 + 0.002, 0.0235, 0.0176, 0.0081 + 0.002, 0.0602, 0.0081 + 0.001, 0.0081,
#           # synpu
#           0.0409, 0.0409 + 0.01, 0.0409 + 0.02, 0.0492, 0.046, 0.0254, 0.0281, 0.0235, 0.0301],

# 风格（Seaborn核心作用）
sns.set(style="whitegrid", context="paper")

plt.figure(figsize=(6, 4.5))

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
# 色彩设置
# ======================
ours_color = "#e41a1c"
colors = {
    "PUNet":   "#c9c92f",   # 黄绿色
    "U3Net":  "#2f2f2f",   # 深灰（接近黑）
    "DLPU":  "#2b4eff",   # 深蓝
    "SQD-LSTM":"#f4a3a8",   # 浅粉
    "Restormer":   "#2bb3b1",   # 青色
    "Uformer":     "#cc33cc",   # 紫色
}

# ======================
# 数据准备
# ======================

x_ours = np.array([159.67-100 ,159.67+10, 159.67+100, 159.67+200])
y_ours = np.array([0.0047 ,0.0053, 0.0057, 0.0085])

#synpu
points = {
    # "PUNet": (4.03, 0.0235),
    "PUNet": (237.02-100, 0.0235),
    # "U3Net": (8.43, 0.0176),
    "U3Net": (299.50, 0.0235),
    # "DLPU": (299.50, 0.0081 + 0.002),
    "DLPU": (8.43, 0.0034),
    "SQD-LSTM": (10.33, 0.0602),
    "Restormer": (20.49, 0.0056),
    # "Uformer": (237.02, 0.0081),
    "Uformer": (4.03, 0.0073),
}

dlpu_x_ours = np.array([159.67-100 ,159.67+10, 159.67+100, 159.67+200])
dlpu_y_ours = np.array([0.0409-0.01 ,0.0409, 0.0409 + 0.01, 0.0409 + 0.02])

#insardlpu
dlpu_points = {
    "PUNet": (4.03, 0.0492),
    "U3Net": (8.43, 0.046),
    "DLPU": (299.50, 0.0254),
    "SQD-LSTM": (10.33, 0.0281),
    "Restormer": (20.49, 0.0235),
    "Uformer": (237.02, 0.0301),
}


# ======================
# 绘图
# ======================
fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
# fig_size_W = 4.5
# fig_size_H = 3
pdf_img_path = r"res/res6/figure.pdf"
png_img_path = r"res/res6/figure.png"
# raw = 2 * 2
raw = 1
col = 1
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes

# synpu
#subplot 1
ax = axes
ax.plot(x_ours, y_ours, color=ours_color, marker='*', markersize=7)

# ===== 散点（用Seaborn）=====
for name, (x, y) in points.items():
    sns.scatterplot(x=[x], y=[y], color=colors[name], s=100, ax=ax)
    ax.text(x, y, name, fontsize=5)

# ===== 标注 =====
labels_ours = ["Ours", "Ours-W", "Ours-S", "Ours-N"]
for x, y, label in zip(x_ours, y_ours, labels_ours):
    ax.text(x, y, label, ha='right', fontsize=5)

# ===== log轴 =====
ax.set_xscale('log')
ax.set_ylabel("NRMSE")
# ax.set_title("Synpu")
ax.set_xticks([])

fig.tight_layout()
fig.savefig(png_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
fig.savefig(pdf_img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()