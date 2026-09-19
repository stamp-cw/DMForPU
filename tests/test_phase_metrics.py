import numpy as np
import torch

from utils.phase_metrics import (
    au_metrics_numpy,
    au_metrics_torch,
    integer_aligned_au,
    range_aligned_au,
    raw_au,
    u3_aligned_metrics_numpy,
    u3_aligned_metrics_torch,
)


def test_u3_alignment_removes_positive_scale_and_offset():
    axis = np.linspace(-2.0, 3.0, 128, dtype=np.float32)
    target = (axis[:, None] + 0.3 * np.sin(axis[None, :] * 4))[None]
    prediction = 2.5 * target + 7.0
    metrics = u3_aligned_metrics_numpy(prediction, target)
    assert metrics["u3_aligned_mae"][0] < 1e-6
    assert metrics["u3_aligned_rmse"][0] < 1e-6
    assert metrics["u3_aligned_nrmse"][0] < 1e-6
    assert abs(metrics["u3_aligned_ssim"][0] - 1.0) < 1e-6


def test_numpy_and_torch_u3_metrics_agree():
    rng = np.random.default_rng(7)
    target = rng.normal(size=(3, 128, 128)).astype(np.float32)
    prediction = target + 0.15 * rng.normal(size=target.shape).astype(np.float32)
    numpy_metrics = u3_aligned_metrics_numpy(prediction, target)
    torch_metrics = u3_aligned_metrics_torch(
        torch.from_numpy(prediction)[:, None], torch.from_numpy(target)[:, None])
    for key in numpy_metrics:
        np.testing.assert_allclose(
            numpy_metrics[key], torch_metrics[key].detach().cpu().numpy(), rtol=2e-4, atol=2e-5)


def test_au_definitions_and_alignment_cases():
    axis = np.linspace(0.0, 20.0, 128, dtype=np.float32)
    target = (axis[:, None] + 0.2 * axis[None, :])[None]

    assert raw_au(target, target)["au"][0] == 100.0
    assert integer_aligned_au(target, target)["au"][0] == 100.0
    assert range_aligned_au(target, target)["au"][0] > 99.9

    integer_shift = target + 2 * np.pi
    assert integer_aligned_au(integer_shift, target)["au"][0] == 100.0
    assert raw_au(integer_shift, target)["au"][0] < 100.0

    real_shift = target + 1.0
    assert integer_aligned_au(real_shift, target)["au"][0] < 100.0

    affine = 0.5 * target + 10.0
    assert range_aligned_au(affine, target)["au"][0] > 99.9
    assert raw_au(affine, target)["au"][0] < 100.0
    assert integer_aligned_au(affine, target)["au"][0] < 100.0

    local_slip = target.copy()
    local_slip[:, :, 3 * target.shape[-1] // 5:] += 2 * np.pi
    assert integer_aligned_au(local_slip, target)["au"][0] < 100.0
    assert range_aligned_au(local_slip, target)["au"][0] < 100.0


def test_numpy_and_torch_au_metrics_agree_and_support_bchw():
    rng = np.random.default_rng(12)
    target = rng.normal(size=(2, 64, 64)).astype(np.float32)
    prediction = target + 0.2 * rng.normal(size=target.shape).astype(np.float32)
    numpy_metrics = au_metrics_numpy(prediction, target)
    torch_metrics = au_metrics_torch(
        torch.from_numpy(prediction)[:, None], torch.from_numpy(target)[:, None])
    for key in numpy_metrics:
        np.testing.assert_allclose(
            numpy_metrics[key], torch_metrics[key].cpu().numpy(), rtol=0, atol=1e-5)
