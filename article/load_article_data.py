import scipy.io as sio
import torch
from spectral.io import envi
import numpy as np

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


def load_article_data():
    # gt
    gt_mat_path = r"data/gt/gt.mat"
    gt_mat = sio.loadmat(gt_mat_path)['gt']

    # keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
    #dlpu
    dlpu_pred_batch_path = "data/dlpu/samples_0_1.pt"
    dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
    dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']
    dlpu_pred = _to_numpy_2d(dlpu_pred)
    # dlpu_diff = gt_mat - _to_numpy_2d(dlpu_pred)

    #punet
    punet_pred_batch_path = "data/punet/samples_0_1.pt"
    punet_pred_batch_pt = torch.load(punet_pred_batch_path)
    punet_pred = punet_pred_batch_pt['pred_unwrapped']
    # punet_diff = gt_mat - _to_numpy_2d(punet_pred)
    punet_pred = _to_numpy_2d(punet_pred)

    #restormer
    restormer_pred_batch_path = "data/restormer/samples_0_1.pt"
    restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
    restormer_pred = restormer_pred_batch_pt['pred_unwrapped']
    # restormer_diff = gt_mat - _to_numpy_2d(restormer_pred)
    restormer_pred = _to_numpy_2d(restormer_pred)

    #snaphu
    snaphu_pred_path = "data/snaphu/000001.hdr"
    snaphu_mat = envi.open(snaphu_pred_path)
    snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)

    #sqd_lstm
    sqd_lstm_pred_batch_path = "data/sqd_lstm/samples_0_1.pt"
    sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
    sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']
    # sqd_lstm_diff = gt_mat - _to_numpy_2d(sqd_lstm_pred)
    sqd_lstm_pred = _to_numpy_2d(sqd_lstm_pred)

    #u3net
    u3net_pred_batch_path = "data/u3net/samples_0_1.pt"
    u3net_pred_batch_pt = torch.load(u3net_pred_batch_path)
    u3net_pred = u3net_pred_batch_pt['pred_unwrapped']
    # u3net_diff = gt_mat - _to_numpy_2d(u3net_pred)
    u3net_pred = _to_numpy_2d(u3net_pred)

    #uformer
    uformer_pred_batch_path = "data/uformer/samples_0_1.pt"
    uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
    uformer_pred = uformer_pred_batch_pt['pred_unwrapped']
    # uformer_diff = gt_mat - _to_numpy_2d(uformer_pred)
    uformer_pred = _to_numpy_2d(uformer_pred)

    #ours
    ours_pred_batch_path = "data/ours/samples_0_1.pt"
    ours_pred_batch_pt = torch.load(ours_pred_batch_path)
    ours_pred = ours_pred_batch_pt['pred_unwrapped']
    # ours_diff = gt_mat - _to_numpy_2d(ours_pred)
    ours_pred = _to_numpy_2d(ours_pred)

    return {
        "gt": gt_mat,
        "dlpu_pred": dlpu_pred,
        "punet_pred": punet_pred,
        "restormer_pred": restormer_pred,
        "snaphu_pred": snaphu_pred,
        "sqd_lstm_pred": sqd_lstm_pred,
        "u3net_pred": u3net_pred,
        "uformer_pred": uformer_pred,
        "ours_pred": ours_pred,
    }

def load_article_data_dlpu():
    # gt
    gt_mat_path = r"data_dlpu/gt/gt.mat"
    gt_mat = sio.loadmat(gt_mat_path)['output']

    # keys: wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped
    #dlpu
    dlpu_pred_batch_path = "data_dlpu/dlpu/samples_0_1.pt"
    dlpu_pred_batch_pt = torch.load(dlpu_pred_batch_path)
    dlpu_pred = dlpu_pred_batch_pt['pred_unwrapped']
    dlpu_pred = _to_numpy_2d(dlpu_pred)
    # dlpu_diff = gt_mat - _to_numpy_2d(dlpu_pred)

    #punet
    punet_pred_batch_path = "data_dlpu/punet/samples_0_1.pt"
    punet_pred_batch_pt = torch.load(punet_pred_batch_path)
    punet_pred = punet_pred_batch_pt['pred_unwrapped']
    # punet_diff = gt_mat - _to_numpy_2d(punet_pred)
    punet_pred = _to_numpy_2d(punet_pred)

    #restormer
    restormer_pred_batch_path = "data_dlpu/restormer/samples_0_1.pt"
    restormer_pred_batch_pt = torch.load(restormer_pred_batch_path)
    restormer_pred = restormer_pred_batch_pt['pred_unwrapped']
    # restormer_diff = gt_mat - _to_numpy_2d(restormer_pred)
    restormer_pred = _to_numpy_2d(restormer_pred)

    #snaphu
    snaphu_pred_path = "data_dlpu/snaphu/000001.hdr"
    snaphu_mat = envi.open(snaphu_pred_path)
    snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)

    #sqd_lstm
    sqd_lstm_pred_batch_path = "data_dlpu/sqd_lstm/samples_0_1.pt"
    sqd_lstm_pred_batch_pt = torch.load(sqd_lstm_pred_batch_path)
    sqd_lstm_pred = sqd_lstm_pred_batch_pt['pred_unwrapped']
    # sqd_lstm_diff = gt_mat - _to_numpy_2d(sqd_lstm_pred)
    sqd_lstm_pred = _to_numpy_2d(sqd_lstm_pred)

    #u3net
    u3net_pred_batch_path = "data_dlpu/u3net/samples_0_1.pt"
    u3net_pred_batch_pt = torch.load(u3net_pred_batch_path)
    u3net_pred = u3net_pred_batch_pt['pred_unwrapped']
    # u3net_diff = gt_mat - _to_numpy_2d(u3net_pred)
    u3net_pred = _to_numpy_2d(u3net_pred)

    #uformer
    uformer_pred_batch_path = "data_dlpu/uformer/samples_0_1.pt"
    uformer_pred_batch_pt = torch.load(uformer_pred_batch_path)
    uformer_pred = uformer_pred_batch_pt['pred_unwrapped']
    # uformer_diff = gt_mat - _to_numpy_2d(uformer_pred)
    uformer_pred = _to_numpy_2d(uformer_pred)

    #ours
    ours_pred_batch_path = "data_dlpu/ours/samples_0_1.pt"
    ours_pred_batch_pt = torch.load(ours_pred_batch_path)
    ours_pred = ours_pred_batch_pt['pred_unwrapped']
    # ours_diff = gt_mat - _to_numpy_2d(ours_pred)
    ours_pred = _to_numpy_2d(ours_pred)

    return {
        "gt": gt_mat,
        "dlpu_pred": dlpu_pred,
        "punet_pred": punet_pred,
        "restormer_pred": restormer_pred,
        "snaphu_pred": snaphu_pred,
        "sqd_lstm_pred": sqd_lstm_pred,
        "u3net_pred": u3net_pred,
        "uformer_pred": uformer_pred,
        "ours_pred": ours_pred,
    }



def load_article_ours_snr_data():
    #ours
    ours_0db_batch_path = "data_snr/ours_snr/0db/samples_0_1.pt"
    ours_0db_batch_pt = torch.load(ours_0db_batch_path)
    ours_0db = ours_0db_batch_pt['pred_unwrapped']
    ours_pred = _to_numpy_2d(ours_0db)

    ours_5db_batch_path = "data_snr/ours_snr/5db/samples_0_1.pt"
    ours_5db_batch_pt = torch.load(ours_5db_batch_path)
    ours_5db = ours_5db_batch_pt['pred_unwrapped']
    ours_5db = _to_numpy_2d(ours_5db)

    ours_10db_batch_path = "data_snr/ours_snr/10db/samples_0_1.pt"
    ours_10db_batch_pt = torch.load(ours_10db_batch_path)
    ours_10db = ours_10db_batch_pt['pred_unwrapped']
    ours_10db = _to_numpy_2d(ours_10db)

    ours_20db_batch_path = "data_snr/ours_snr/20db/samples_0_1.pt"
    ours_20db_batch_pt = torch.load(ours_20db_batch_path)
    ours_20db = ours_20db_batch_pt['pred_unwrapped']
    ours_20db = _to_numpy_2d(ours_20db)

    ours_30db_batch_path = "data_snr/ours_snr/30db/samples_0_1.pt"
    ours_30db_batch_pt = torch.load(ours_30db_batch_path)
    ours_30db = ours_30db_batch_pt['pred_unwrapped']
    ours_30db = _to_numpy_2d(ours_30db)

    return {
        "0db": ours_pred,
        "5db": ours_5db,
        "10db": ours_10db,
        "20db": ours_20db,
        "30db": ours_30db,
    }

def load_article_snr_data(name, snr):
    # if name=="dlpu" or name =="restormer" or name=="ours":
    # if name=="ours":
    #     name = "punet"
    if name == "snaphu":
        snaphu_pred_path = f"data_snr/{name}_snr/{snr}db/000001.hdr"
        snaphu_mat = envi.open(snaphu_pred_path)
        snaphu_pred = snaphu_mat.load()[:,:,0].squeeze(-1)
        return snaphu_pred
    elif name == "wrapped":
        # wrapped_mat_path = f"data_snr/{name}_snr/{snr}db/wrapped.mat"
        # wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
        wrapped_mat_path = f"data_snr/ours_snr/{snr}db/samples_0_1.pt"
        pred_batch_pt = torch.load(wrapped_mat_path)
        wrapped_mat = pred_batch_pt['wrapped']
        wrapped_mat = _to_numpy_2d(wrapped_mat)
        return wrapped_mat
    else:
        pred_batch_path = f"data_snr/{name}_snr/{snr}db/samples_0_1.pt"
        pred_batch_pt = torch.load(pred_batch_path)
        pred = pred_batch_pt['pred_unwrapped']
        # pred = _to_numpy_2d(pred)
        return pred

def load_article_data_idx(name, idx):
    # if name=="dlpu" or name =="restormer":
    #     name = "punet"
    if name == "wrapped":
        wrapped_mat_path = f"data/{name}/00000{idx}.mat"
        wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
        return wrapped_mat
    elif name == "gt":
        wrapped_mat_path = f"data/{name}/00000{idx}.mat"
        wrapped_mat = sio.loadmat(wrapped_mat_path)['gt']
        return wrapped_mat
    else:
        if name == "ours":
            pred_batch_path = f"data/{name}/samples_{idx-1}_{idx}.pt"
        else:
            pred_batch_path = f"data/{name}/samples_{idx*2-2}_{idx*2-1}.pt"
        pred_batch_pt = torch.load(pred_batch_path)
        pred = pred_batch_pt['pred_unwrapped']
        # pred = _to_numpy_2d(pred)
        return pred


def load_article_data_dlpu_idx(name, idx):
    # if name == "ours":
    #     name = "punet"
    if name == "wrapped":
        wrapped_mat_path = f"data_dlpu/{name}/00000{idx}.mat"
        wrapped_mat = sio.loadmat(wrapped_mat_path)['input']
        return wrapped_mat
    elif name == "gt":
        wrapped_mat_path = f"data_dlpu/{name}/00000{idx}.mat"
        wrapped_mat = sio.loadmat(wrapped_mat_path)['output']
        return wrapped_mat
    else:
        if name == "ours":
            pred_batch_path = f"data_dlpu/{name}/samples_{idx-1}_{idx}.pt"
        else:
            pred_batch_path = f"data_dlpu/{name}/samples_{idx*2-2}_{idx*2-1}.pt"
        pred_batch_pt = torch.load(pred_batch_path)
        pred = pred_batch_pt['pred_unwrapped']
        # pred = _to_numpy_2d(pred)
        return pred