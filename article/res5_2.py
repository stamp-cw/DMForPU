from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch

def plot_3d_surface(M, title=None, figsize=(3.5, 2.5),fig_dpi=600, img_path=None):
    # 生成坐标网格
    x = np.arange(M.shape[1])
    y = np.arange(M.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, Y, M,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )
    if title is not None:
        ax.set_title(title)
    # ax.set_xlabel("X")
    # ax.set_zlim(zmin, zmax)
    # fig = plt.figure(figsize=(7, 3))
    # for i, M in enumerate([M1, M2]):
    #     ax = fig.add_subplot(1, 2, i+1, projection="3d")

    # ax.set_ylabel("Y")
    # ax.set_zlabel("Value")
    ax.view_init(elev=30, azim=135)
    plt.tight_layout()
    if img_path is not None:
        plt.savefig(img_path, bbox_inches="tight", dpi=fig_dpi, pad_inches=0)
    plt.show()


def _to_numpy_2d(x: torch.Tensor):
    return x.detach().cpu().squeeze().numpy()

gt_mat_path = r"data/gt/gt.mat"
gt_mat = sio.loadmat(gt_mat_path)['gt']

punet_pred_batch_path = "data/punet/samples_0_1.pt"
punet_pred_batch_pt = torch.load(punet_pred_batch_path)
punet_pred = punet_pred_batch_pt['pred_unwrapped']
punet_diff = gt_mat - _to_numpy_2d(punet_pred)

sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
# keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']
sqd_lstm_diff = gt_mat - _to_numpy_2d(sqd_lstm_pred)

restormer_pred_batch_path = "data/restormer/samples_0_1.pt"
restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
restormer_pred = restormer_pred_batch_pt['pred_unwrapped']
restormer_diff = gt_mat - _to_numpy_2d(restormer_pred)

uformer_pred_batch_path = "data/uformer/samples_0_1.pt"
uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
uformer_pred = uformer_pred_batch_pt['pred_unwrapped']
uformer_diff = gt_mat - _to_numpy_2d(uformer_pred)

ours_pred_batch_path = "data/ours/samples_0_1.pt"
ours_pred_batch_pt = torch.load(ours_pred_batch_path)
ours_pred = ours_pred_batch_pt['pred_unwrapped']
ours_diff = gt_mat - _to_numpy_2d(ours_pred)

fig_size_W = 3.5
fig_size_H = 2.5
fig_dpi = 600

img_path = r"res/res9/3d_punet_diff.png"
plot_3d_surface(punet_diff, title="PUnet Diff", figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_sqd_diff.png"
plot_3d_surface(sqd_lstm_diff, title="SQD-LSTM Diff", figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_restormer_diff.png"
plot_3d_surface(restormer_diff, title="Restormer Diff",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_uformer_diff.png"
plot_3d_surface(uformer_diff, title="Uformer Diff",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_ours_diff.png"
plot_3d_surface(ours_diff, title="Ours Diff",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)




