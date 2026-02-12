# unwrapped diff矩阵对比折线图

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
import torch

from article.res4 import img_path_dir


def plot_multi_matrix_distribution(
        matrices,
        labels,
        bin_width=0.1,
        normalize=False,
        figsize=(3.5, 2.5),
        fig_dpi = 600,
        img_path = None
):
    """
    matrices: list of 2D numpy arrays
    labels:   list of str, same length as matrices
    """

    # 1. 展平所有矩阵
    data_list = [M.flatten() for M in matrices]

    # 2. 统一 min / max（非常关键）
    global_min = min(d.min() for d in data_list)
    global_max = max(d.max() for d in data_list)

    bins = np.arange(global_min, global_max + bin_width, bin_width)

    plt.figure(figsize=figsize)

    # 3. 对每个矩阵画一条折线
    for data, label in zip(data_list, labels):
        counts, edges = np.histogram(data, bins=bins)

        if normalize:
            counts = counts / counts.sum()

        bin_centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(bin_centers, counts, label=label, linewidth=2)

    # 4. 画图细节
    plt.xlabel("Rad Value")
    plt.ylabel("Probability" if normalize else "Count")
    plt.legend(frameon=False)
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

img_path = f"res/res5/diff_distribution.pdf"
plot_multi_matrix_distribution(
    matrices=[punet_diff, sqd_lstm_diff, restormer_diff, uformer_diff, ours_diff],
    labels=["PUNet", "SQD-LSTM", "Restormer", "Uformer", "Ours"],
    bin_width=0.1,
    # normalize=True,
    normalize=False,
    figsize= (fig_size_W, fig_size_H),
    fig_dpi = fig_dpi,
    img_path= img_path
)
