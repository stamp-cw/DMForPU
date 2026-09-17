import unittest

import numpy as np

from traditional.phase_unwrapping import METHODS, wrap_phase


class TraditionalPhaseUnwrappingTest(unittest.TestCase):
    def test_methods_recover_smooth_phase_up_to_global_two_pi_offset(self):
        row, column = np.mgrid[:32, :40]
        phase = 0.45 * row + 0.37 * column + 0.003 * row * column
        wrapped = wrap_phase(phase)

        for name, method in METHODS.items():
            with self.subTest(method=name):
                prediction = method(wrapped)
                global_k = np.rint(np.median((phase - prediction) / (2 * np.pi)))
                aligned = prediction + global_k * 2 * np.pi
                np.testing.assert_allclose(aligned, phase, atol=1e-8)

    def test_methods_are_label_free_and_preserve_shape(self):
        rng = np.random.default_rng(42)
        wrapped = rng.uniform(-np.pi, np.pi, size=(17, 19))

        for name, method in METHODS.items():
            with self.subTest(method=name):
                prediction = method(wrapped)
                self.assertEqual(prediction.shape, wrapped.shape)
                self.assertTrue(np.isfinite(prediction).all())

    def test_path_following_methods_preserve_wrapped_observation(self):
        rng = np.random.default_rng(42)
        wrapped = rng.uniform(-np.pi, np.pi, size=(17, 19))

        for name in ("itoh", "quality_guided_mst"):
            with self.subTest(method=name):
                prediction = METHODS[name](wrapped)
                np.testing.assert_allclose(wrap_phase(prediction), wrapped, atol=1e-8)

    def test_non_2d_input_is_rejected(self):
        for name, method in METHODS.items():
            with self.subTest(method=name):
                with self.assertRaises(ValueError):
                    method(np.zeros((1, 4, 4)))
