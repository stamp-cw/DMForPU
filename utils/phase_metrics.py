"""Shared phase-unwrapping metrics used by learned and traditional methods."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


_SSIM_RADIUS = 5
_SSIM_SIGMA = 1.5
_TWO_PI = 2 * np.pi


def _numpy_bhw(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 2:
        value = value[None]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"expected [H,W], [B,H,W], or [B,1,H,W], got {value.shape}")
    return value if np.issubdtype(value.dtype, np.floating) else value.astype(np.float64)


def _torch_bhw(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        value = value[None]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError(f"expected [H,W], [B,H,W], or [B,1,H,W], got {tuple(value.shape)}")
    return value.float()


def _compute_au(aligned_pred, target, eta: float = 0.05):
    """Shared AU decision rule; returns one percentage value per image."""
    if not 0 <= eta:
        raise ValueError("eta must be non-negative")
    if torch.is_tensor(aligned_pred):
        aligned_pred, target = _torch_bhw(aligned_pred), _torch_bhw(target)
        target_min = target.flatten(1).amin(1)[:, None, None]
        tolerance = (target - target_min).abs() * eta
        return ((aligned_pred - target).abs() <= tolerance).float().flatten(1).mean(1) * 100
    aligned_pred, target = _numpy_bhw(aligned_pred), _numpy_bhw(target)
    target_min = target.min(axis=(1, 2), keepdims=True)
    tolerance = np.abs(target - target_min) * eta
    return np.mean(np.abs(aligned_pred - target) <= tolerance, axis=(1, 2)) * 100


def raw_au(prediction, target, eta: float = 0.05) -> dict:
    """Raw AU without phase alignment."""
    aligned = _torch_bhw(prediction) if torch.is_tensor(prediction) else _numpy_bhw(prediction)
    return {"au": _compute_au(aligned, target, eta), "aligned_pred": aligned}


def integer_aligned_au(prediction, target, eta: float = 0.05) -> dict:
    """AU after independently removing a global integer 2*pi offset per image."""
    if torch.is_tensor(prediction):
        prediction, target = _torch_bhw(prediction), _torch_bhw(target)
        k = torch.round(torch.quantile((prediction - target).flatten(1), 0.5, dim=1) / _TWO_PI)
        aligned = prediction - _TWO_PI * k[:, None, None]
    else:
        prediction, target = _numpy_bhw(prediction), _numpy_bhw(target)
        two_pi = np.asarray(_TWO_PI, dtype=prediction.dtype)
        k = np.round(np.median(prediction - target, axis=(1, 2)) / two_pi)
        aligned = prediction - two_pi * k[:, None, None]
    return {"au": _compute_au(aligned, target, eta), "aligned_pred": aligned, "k": k}


def range_aligned_au(prediction, target, eta: float = 0.05,
                     eps: float = 1e-8) -> dict:
    """AU after U3Net-style min-max range alignment; this is not an official U3Net AU."""
    if torch.is_tensor(prediction):
        prediction, target = _torch_bhw(prediction), _torch_bhw(target)
        pmin = prediction.flatten(1).amin(1)[:, None, None]
        pmax = prediction.flatten(1).amax(1)[:, None, None]
        tmin = target.flatten(1).amin(1)[:, None, None]
        tmax = target.flatten(1).amax(1)[:, None, None]
    else:
        prediction, target = _numpy_bhw(prediction), _numpy_bhw(target)
        pmin = prediction.min(axis=(1, 2), keepdims=True)
        pmax = prediction.max(axis=(1, 2), keepdims=True)
        tmin = target.min(axis=(1, 2), keepdims=True)
        tmax = target.max(axis=(1, 2), keepdims=True)
    aligned = (prediction - pmin) / (pmax - pmin + eps) * (tmax - tmin) + tmin
    return {"au": _compute_au(aligned, target, eta), "aligned_pred": aligned}


def au_metrics_numpy(prediction: np.ndarray, target: np.ndarray,
                     eta: float = 0.05) -> dict[str, np.ndarray]:
    return {
        "raw_au": raw_au(prediction, target, eta)["au"],
        "integer_aligned_au": integer_aligned_au(prediction, target, eta)["au"],
        "range_aligned_au": range_aligned_au(prediction, target, eta)["au"],
    }


def au_metrics_torch(prediction: torch.Tensor, target: torch.Tensor,
                     eta: float = 0.05) -> dict[str, torch.Tensor]:
    return {
        "raw_au": raw_au(prediction, target, eta)["au"],
        "integer_aligned_au": integer_aligned_au(prediction, target, eta)["au"],
        "range_aligned_au": range_aligned_au(prediction, target, eta)["au"],
    }


def u3_align_numpy(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply U3Net's per-image min-max affine alignment to a prediction batch."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim == 2:
        prediction, target = prediction[None], target[None]
    axes = tuple(range(1, prediction.ndim))
    pmin = prediction.min(axis=axes, keepdims=True)
    pspan = prediction.max(axis=axes, keepdims=True) - pmin
    tmin = target.min(axis=axes, keepdims=True)
    tspan = target.max(axis=axes, keepdims=True) - tmin
    return (prediction - pmin) / np.maximum(pspan, 1e-12) * tspan + tmin


def _ssim_numpy(prediction: np.ndarray, target: np.ndarray, data_range: np.ndarray) -> np.ndarray:
    """Wang et al. SSIM with an 11x11 Gaussian window, returned per image."""
    values = []
    crop = _SSIM_RADIUS
    for x, y, span in zip(prediction, target, data_range):
        ux = gaussian_filter(x, _SSIM_SIGMA, radius=crop, mode="reflect")
        uy = gaussian_filter(y, _SSIM_SIGMA, radius=crop, mode="reflect")
        vx = gaussian_filter(x * x, _SSIM_SIGMA, radius=crop, mode="reflect") - ux * ux
        vy = gaussian_filter(y * y, _SSIM_SIGMA, radius=crop, mode="reflect") - uy * uy
        vxy = gaussian_filter(x * y, _SSIM_SIGMA, radius=crop, mode="reflect") - ux * uy
        c1, c2 = (0.01 * span) ** 2, (0.03 * span) ** 2
        score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
            (ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-18)
        values.append(float(score[crop:-crop, crop:-crop].mean()))
    return np.asarray(values, dtype=np.float64)


def u3_aligned_metrics_numpy(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """U3Net-style aligned MAE/RMSE/NRMSE and aligned SSIM, per image."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim == 2:
        prediction, target = prediction[None], target[None]
    aligned = u3_align_numpy(prediction, target)
    axes = tuple(range(1, aligned.ndim))
    error = aligned - target
    span = np.maximum(target.max(axis=axes) - target.min(axis=axes), 1e-12)
    rmse = np.sqrt(np.mean(error * error, axis=axes))
    return {
        "u3_aligned_mae": np.mean(np.abs(error), axis=axes),
        "u3_aligned_rmse": rmse,
        "u3_aligned_nrmse": rmse / span,
        "u3_aligned_ssim": _ssim_numpy(aligned, target, span),
    }


def u3_align_torch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PyTorch equivalent of :func:`u3_align_numpy` for BCHW tensors."""
    prediction, target = prediction.float(), target.float()
    pmin = prediction.flatten(1).amin(1)[:, None, None, None]
    pspan = prediction.flatten(1).amax(1)[:, None, None, None] - pmin
    tmin = target.flatten(1).amin(1)[:, None, None, None]
    tspan = target.flatten(1).amax(1)[:, None, None, None] - tmin
    return (prediction - pmin) / pspan.clamp_min(1e-12) * tspan + tmin


def _ssim_torch(prediction: torch.Tensor, target: torch.Tensor,
                data_range: torch.Tensor) -> torch.Tensor:
    coords = torch.arange(-_SSIM_RADIUS, _SSIM_RADIUS + 1,
                          device=prediction.device, dtype=prediction.dtype)
    kernel_1d = torch.exp(-(coords * coords) / (2 * _SSIM_SIGMA ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :])[None, None]
    ux, uy = F.conv2d(prediction, kernel), F.conv2d(target, kernel)
    vx = F.conv2d(prediction * prediction, kernel) - ux * ux
    vy = F.conv2d(target * target, kernel) - uy * uy
    vxy = F.conv2d(prediction * target, kernel) - ux * uy
    vx, vy = vx.clamp_min(0), vy.clamp_min(0)
    span = data_range[:, None, None, None]
    c1, c2 = (0.01 * span) ** 2, (0.03 * span) ** 2
    score = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2) + 1e-12)
    return score.flatten(1).mean(1)


def u3_aligned_metrics_torch(prediction: torch.Tensor,
                             target: torch.Tensor) -> dict[str, torch.Tensor]:
    """U3Net-style aligned MAE/RMSE/NRMSE and aligned SSIM, per image."""
    target = target.float()
    aligned = u3_align_torch(prediction, target)
    error = aligned - target
    span = (target.flatten(1).amax(1) - target.flatten(1).amin(1)).clamp_min(1e-12)
    rmse = error.square().flatten(1).mean(1).sqrt()
    return {
        "u3_aligned_mae": error.abs().flatten(1).mean(1),
        "u3_aligned_rmse": rmse,
        "u3_aligned_nrmse": rmse / span,
        "u3_aligned_ssim": _ssim_torch(aligned, target, span),
    }
