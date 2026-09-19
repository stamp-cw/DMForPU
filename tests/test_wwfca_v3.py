import unittest

import torch

from model.fdunet.wwfca_v3 import MidBlockWithWWFCAV3, WWFCAv3UNet


class TestWWFCAV3(unittest.TestCase):
    def test_sigma_encoding_is_continuous_at_clean(self):
        sigma = torch.tensor([0.0, 1e-6, 0.1, 1.0])
        encoded = MidBlockWithWWFCAV3.encode_sigma(sigma)
        self.assertEqual(float(encoded[0]), 0.0)
        self.assertTrue(torch.all(encoded[1:] > encoded[:-1]))
        self.assertLess(float(encoded[1]), 2e-6)

    def test_disabled_and_enabled_models_start_identically(self):
        torch.manual_seed(7)
        off = WWFCAv3UNet(False, widths=(32, 32, 32, 32), layers=1).eval()
        torch.manual_seed(7)
        on = WWFCAv3UNet(True, widths=(32, 32, 32, 32), layers=1).eval()
        x = torch.randn(2, 2, 32, 32)
        t = torch.tensor([20, 700])
        sigma = torch.tensor([0.0, 1.0])
        with torch.no_grad():
            a = off(x, t, sigma).sample
            b = on(x, t, sigma).sample
        self.assertTrue(torch.equal(a, b))

    def test_warmup_scale_keeps_exact_baseline_after_gate_changes(self):
        model = WWFCAv3UNet(True, widths=(32, 32, 32, 32), layers=1).eval()
        with torch.no_grad():
            model.unet.mid_block.gate[-1].bias.fill_(10.0)
        model.set_gate_scale(0.0)
        x = torch.randn(1, 2, 32, 32)
        t = torch.tensor([50])
        sigma = torch.tensor([0.2])
        with torch.no_grad():
            actual = model(x, t, sigma).sample
            expected = model.unet(x, t).sample
        self.assertTrue(torch.equal(actual, expected))

    def test_gate_is_hard_bounded(self):
        model = WWFCAv3UNet(True, widths=(32, 32, 32, 32), layers=1).eval()
        with torch.no_grad():
            model.unet.mid_block.gate[-1].bias.fill_(100.0)
        model.set_gate_scale(1.0)
        with torch.no_grad():
            model(torch.randn(1, 2, 32, 32), torch.tensor([500]), torch.tensor([0.0]))
        diagnostics = model.gate_diagnostics()
        self.assertLessEqual(diagnostics["gate_max_abs"], 0.100001)

    def test_gate_and_frequency_branch_receive_gradients(self):
        model = WWFCAv3UNet(True, widths=(32, 32, 32, 32), layers=1)
        model.set_gate_scale(0.5)
        x = torch.randn(2, 2, 32, 32)
        y = model(x, torch.tensor([100, 600]), torch.tensor([0.0, 0.5])).sample
        (y.square().mean() + 0.01 * model.gate_regularization()).backward()
        self.assertGreater(float(model.unet.mid_block.gate[-1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
