import numpy as np

from traditional.phase_unwrapping import unwrap_dct_schofield, wrap_phase


def test_dct_unwraps_a_sub_nyquist_plane_up_to_constant():
    yy, xx = np.mgrid[:32, :32]
    target = 0.31 * xx + 0.27 * yy - 7.0
    prediction = unwrap_dct_schofield(wrap_phase(target))
    error = prediction - target
    error -= error.mean()
    assert np.max(np.abs(error)) < 1e-10


def test_dct_output_rewraps_exactly_to_observation():
    rng = np.random.default_rng(7)
    wrapped = rng.uniform(-np.pi, np.pi, (19, 23))
    prediction = unwrap_dct_schofield(wrapped)
    circular_error = wrap_phase(prediction - wrapped)
    assert np.max(np.abs(circular_error)) < 1e-12
