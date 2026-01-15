from typing import Dict

import torch
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