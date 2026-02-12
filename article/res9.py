from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch

from article.load_article_data import load_article_data


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

fig_size_W = 3.5
fig_size_H = 2.5
fig_dpi = 600

img_path = r"res/res9/3d_gt.png"
plot_3d_surface(gt, title="GT", figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_punet_pred.png"
plot_3d_surface(punet_pred, title="PUnet Pred", figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_sqd_pred.png"
plot_3d_surface(sqd_lstm_pred, title="SQD-LSTM Pred", figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_restormer_pred.png"
plot_3d_surface(restormer_pred, title="Restormer Pred",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_uformer_pred.png"
plot_3d_surface(uformer_pred, title="Uformer Pred",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)

img_path = r"res/res9/3d_ours_pred.png"
plot_3d_surface(ours_pred, title="Ours Pred",figsize=(fig_size_W, fig_size_H), fig_dpi=fig_dpi, img_path=img_path)




