import numpy as np

from traditional.phase_unwrapping import unwrap_dct_schofield, wrap_phase


def test_dct_cycle_projection_is_stable_at_neumann_boundaries():
    row, column = np.mgrid[:32, :40]
    phase = 0.45 * row + 0.37 * column + 0.003 * row * column
    prediction = unwrap_dct_schofield(wrap_phase(phase))
    global_k = np.rint(np.median((phase - prediction) / (2 * np.pi)))
    np.testing.assert_allclose(prediction + global_k * 2 * np.pi, phase, atol=1e-8)
