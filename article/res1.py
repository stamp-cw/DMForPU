import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
def pi_formatter(x, pos):
    k = x / np.pi

    # 0
    if np.isclose(k, 0):
        return "0"

    # 接近整数倍 π
    if np.isclose(k, round(k)):
        k_int = int(round(k))
        if k_int == 1:
            return r"$\pi$"
        elif k_int == -1:
            return r"$-\pi$"
        else:
            return rf"${k_int}\pi$"

    # 非整数倍：显示数值（可改精度）
    return rf"${x:.2f}$"

def _to_numpy_2d(x: torch.Tensor):
    return x.detach().cpu().squeeze().numpy()

wrapped_mat_path = r"data/wrapped/wrapped.mat"
gt_mat_path = r"data/gt/gt.mat"
wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
gt_mat = sio.loadmat(gt_mat_path)['gt']
ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']
ours_mat = _to_numpy_2d(ours_pred)

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

fig_dpi = 600
fig_size_W = 3.5
fig_size_H = 2.5
img_path = r"res/res1/figure.pdf"
raw = 1
col = 3
fig, axes = plt.subplots(raw, col, figsize=(fig_size_W * col, fig_size_H * raw))
axes = axes.flatten()

# raw 1, col 1
color_norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
ax_0 = axes[0]
im_0 = ax_0.imshow(wrapped_mat, cmap="twilight", norm=color_norm)
# ax_0.set_title("Wrapped")
ax_0.set_xlabel("(a) Wrapped", fontsize=7, labelpad=6)
# ax_0.axis("off")
# ax_0.legend(frameon=False)
cbar = fig.colorbar(im_0, ax=ax_0, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

# raw 1, col 2
color_norm = colors.Normalize(vmin=0)
ax_1 = axes[1]
im_1 = ax_1.imshow(gt_mat, cmap="turbo", norm=color_norm)
# ax_1.set_title("Gt Unwrapped")
ax_1.set_xlabel("(b) Gt", fontsize=7, labelpad=6)
# ax_1.axis("off")
# ax_1.legend(frameon=False)
cbar = fig.colorbar(im_1, ax=ax_1, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

# raw 1, col 3
# color_norm = colors.Normalize(vmin=0)
ax_2 = axes[2]
# im_2 = ax_2.imshow(ours_mat, cmap="turbo", norm=color_norm)
im_2 = ax_2.imshow(ours_mat, cmap="turbo")
# ax_2.set_title("Gt Unwrapped")
ax_2.set_xlabel("(c) Ours", fontsize=7, labelpad=6)
# ax_2.axis("off")
# ax_1.legend(frameon=False)
cbar = fig.colorbar(im_2, ax=ax_2, fraction=0.046, pad=0.04)
cbar.locator = MultipleLocator(np.pi)
cbar.formatter = FuncFormatter(pi_formatter)
cbar.update_ticks()

fig.tight_layout()
fig.savefig(img_path, dpi=fig_dpi, bbox_inches="tight", pad_inches=0)
plt.show()
# plt.close(fig)