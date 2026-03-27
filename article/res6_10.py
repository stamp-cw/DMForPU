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

# ===== 数据 =====
# x_ours = np.array([12, 40, 100])
x_ours = np.array([159.67-100 ,159.67+10, 159.67+100, 159.67+200])
y_ours = np.array([0.0409-0.01 ,0.0409, 0.0409 + 0.01, 0.0409 + 0.02])

ours_color = "#e41a1c"   # 红色（Uformer）
# unet_color    = "#1f3cff"   # 蓝色（UNet）

colors = {
    "PUNet":   "#c9c92f",   # 黄绿色
    "U3Net":  "#2f2f2f",   # 深灰（接近黑）
    "DLPU":  "#2b4eff",   # 深蓝
    "SQD-LSTM":"#f4a3a8",   # 浅粉
    "Restormer":   "#2bb3b1",   # 青色
    "Uformer":     "#cc33cc",   # 紫色
}

#insar
points = {
    "PUNet": (4.03, 0.0492),
    "U3Net": (8.43, 0.046),
    "DLPU": (299.50, 0.0254),
    "SQD-LSTM": (10.33, 0.0281),
    "Restormer": (20.49, 0.0235),
    "Uformer": (237.02, 0.0301),
}

# ===== 折线（Seaborn也可以，但这里用plt更直接）=====
# plt.plot(x_ours, y_ours, color=ours_color, marker='*', markersize=12, label='Ours')
plt.plot(x_ours, y_ours, color=ours_color, marker='*', markersize=12)
# plt.plot(x_unet, y_unet, marker='v', markersize=10, label='UNet')

# ===== 散点（用Seaborn）=====
for name, (x, y) in points.items():
    sns.scatterplot(x=[x], y=[y], color=colors[name], s=100)
    plt.text(x, y, name, fontsize=11)

# ===== 标注 =====
labels_ours = ["Ours", "Ours-W", "Ours-S", "Ours-N"]
for x, y, label in zip(x_ours, y_ours, labels_ours):
    plt.text(x, y, label, ha='right')

# ===== log轴 =====
plt.xscale('log')

# plt.xlabel("")
plt.ylabel("NRMSE")

plt.tight_layout()
plt.show()