import unittest

import torch

from model.fdunet.fdunet_atten import (
    FDBasicTransformerBlock,
    FDTransformer2DModel,
    _haar_dwt2,
    _merge_windows,
    _partition_windows,
)


class WWFCATests(unittest.TestCase):
    def test_window_partition_is_exactly_reversible(self):
        x = torch.arange(2 * 3 * 8 * 12, dtype=torch.float32).reshape(2, 3, 8, 12)
        windows, num_h, num_w = _partition_windows(x, 4)
        restored = _merge_windows(windows, 2, num_h, num_w)
        torch.testing.assert_close(restored, x)

    def test_haar_transform_preserves_energy(self):
        torch.manual_seed(7)
        x = torch.randn(2, 5, 8, 6)
        bands = _haar_dwt2(x)
        self.assertTrue(all(band.shape == (2, 5, 4, 3) for band in bands))
        output_energy = sum(band.square().sum() for band in bands)
        torch.testing.assert_close(output_energy, x.square().sum(), rtol=1e-5, atol=1e-5)

    def test_constant_input_has_no_high_frequency_energy(self):
        ll, lh, hl, hh = _haar_dwt2(torch.ones(1, 2, 8, 8))
        torch.testing.assert_close(ll, torch.full_like(ll, 2.0))
        for band in (lh, hl, hh):
            torch.testing.assert_close(band, torch.zeros_like(band))

    def test_frequency_attention_supports_multiple_rectangular_windows(self):
        block = FDBasicTransformerBlock(
            dim=16,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=48,
        )
        projection_inputs = {}

        def record(name):
            def hook(_module, args):
                projection_inputs[name] = tuple(args[0].shape)
            return hook

        q_hook = block.attn2.to_q.register_forward_pre_hook(record("q"))
        k_hook = block.attn2.to_k.register_forward_pre_hook(record("k"))
        x = torch.randn(2, 8 * 12, 16, requires_grad=True)
        y = block._frequency_cross_attention(x, (8, 12), {})
        q_hook.remove()
        k_hook.remove()
        self.assertEqual(y.shape, x.shape)
        # 8x12 selects M=4: 6 windows/image and 4 wavelet tokens/window.
        self.assertEqual(projection_inputs["q"], (12, 4, 16))
        self.assertEqual(projection_inputs["k"], (12, 4, 48))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_transformer_threads_rectangular_spatial_shape(self):
        model = FDTransformer2DModel(
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=16,
            num_layers=1,
            cross_attention_dim=48,
            norm_num_groups=8,
        )
        x = torch.randn(1, 16, 8, 12, requires_grad=True)
        y = model(x).sample
        self.assertEqual(y.shape, x.shape)
        y.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_frequency_context_dimension_is_rejected(self):
        block = FDBasicTransformerBlock(
            dim=16,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
        )
        with self.assertRaisesRegex(ValueError, "3 \\* feature channels"):
            block._frequency_cross_attention(torch.randn(1, 64, 16), (8, 8), {})


if __name__ == "__main__":
    unittest.main()
