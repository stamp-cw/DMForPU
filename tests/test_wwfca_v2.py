import unittest

import torch

from model.fdunet.fdunet_atten import _haar_dwt2
from model.fdunet.wwfca_v2 import WWFCAHighFromLow, WWFCAv2UNet, haar_idwt2


class TestWWFCAV2(unittest.TestCase):
    def test_haar_round_trip(self):
        x = torch.randn(3, 7, 16, 12)
        self.assertTrue(torch.allclose(haar_idwt2(*_haar_dwt2(x)), x, atol=1e-6, rtol=1e-6))

    def test_frequency_branch_shape_and_gradient(self):
        module = WWFCAHighFromLow(32, heads=4, window_size=8, dropout=0.0)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)
        y = module(x)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_on_and_off_start_from_identical_backbone_output(self):
        torch.manual_seed(12)
        off = WWFCAv2UNet(False, widths=(32, 32, 32, 32), layers=1).eval()
        torch.manual_seed(12)
        on = WWFCAv2UNet(True, widths=(32, 32, 32, 32), layers=1).eval()
        x = torch.randn(2, 2, 32, 32)
        t = torch.tensor([10, 500])
        sigma = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            a = off(x, t, sigma).sample
            b = on(x, t, sigma).sample
        self.assertTrue(torch.equal(a, b))

    def test_zero_gate_receives_gradient(self):
        model = WWFCAv2UNet(True, widths=(32, 32, 32, 32), layers=1)
        x = torch.randn(2, 2, 32, 32)
        y = model(x, torch.tensor([20, 200]), torch.tensor([0.0, 0.5])).sample
        y.square().mean().backward()
        final_gate = model.unet.mid_block.gate[-1]
        self.assertGreater(float(final_gate.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
