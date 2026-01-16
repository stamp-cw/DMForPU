import scipy.io as sio
import torch
import matplotlib.pyplot as plt

wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/train_wrapped/000101.mat')['input']
unwrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat/train_gt_absolute/000101.mat')['input']

wrapped = torch.as_tensor(wrapped).unsqueeze(0)
unwrapped = torch.as_tensor(unwrapped).unsqueeze(0)

def _save_compare_png(path, wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped):
    def _to_numpy_2d(x: torch.Tensor):
        return x.detach().cpu().squeeze().numpy()
    compare_png_path = path
    titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped"]
    imgs = [_to_numpy_2d(wrapped[0]), _to_numpy_2d(gt_unwrapped[0]), _to_numpy_2d(pred_unwrapped[0]), _to_numpy_2d(diff_unwrapped[0])]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cmaps = ["twilight", "turbo", "turbo", "inferno"]
    for ax, img, title, cmap in zip(axes, imgs, titles, cmaps):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(compare_png_path, dpi=200)
    plt.close(fig)

_save_compare_png('compare.png', wrapped, unwrapped, unwrapped - wrapped, unwrapped - wrapped)