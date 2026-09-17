from typing import Dict

import torch

from utils.util import wrap_phase


# def phase_error_metrics(pred_unwrapped: torch.Tensor, gt_unwrapped: torch.Tensor) -> Dict[str, float]:
#     """Compute simple L1/L2 metrics on unwrapped phase."""
#     diff = pred_unwrapped - gt_unwrapped
#     l1 = float(diff.abs().mean())
#     l2 = float(torch.sqrt((diff ** 2).mean()))
#     return {"l1": l1, "rmse": l2}

def l1_metric(pred_unwrapped: torch.Tensor, gt_unwrapped: torch.Tensor):
    """Compute L1 metrics."""
    diff = pred_unwrapped - gt_unwrapped
    l1 = diff.abs().mean()
    return l1

def rmse_metric(pred_unwrapped: torch.Tensor, gt_unwrapped: torch.Tensor):
    """Compute RMSE metrics."""
    diff = pred_unwrapped - gt_unwrapped
    rmse = torch.sqrt((diff ** 2).mean())
    return rmse


def nrmse_metric(pred: torch.Tensor, target: torch.Tensor):
    """Mean per-image RMSE/range, independent of the other images in a batch."""
    target = target.float().flatten(1)
    error = pred.float().flatten(1) - target
    scale = target.amax(1) - target.amin(1)
    return (error.square().mean(1).sqrt() / (scale + 1e-8)).mean()

def wrap_l1_metric(pred_unwrapped: torch.Tensor, gt_unwrapped: torch.Tensor):
    """Compute L1 metrics."""
    pred_wrapped = wrap_phase(pred_unwrapped)
    gt_wrapped = wrap_phase(gt_unwrapped)
    diff = pred_wrapped - gt_wrapped
    l1 = diff.abs().mean()
    return l1

def wrap_rmse_metric(pred_unwrapped: torch.Tensor, gt_unwrapped: torch.Tensor):
    """Compute RMSE metrics."""
    pred_wrapped = wrap_phase(pred_unwrapped)
    gt_wrapped = wrap_phase(gt_unwrapped)
    diff = pred_wrapped - gt_wrapped
    rmse = torch.sqrt((diff ** 2).mean())
    return rmse
