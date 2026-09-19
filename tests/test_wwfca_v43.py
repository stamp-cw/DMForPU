import unittest
import torch

from model.fdunet.wwfca_v43 import ValidateThenRetrieve, WWFCAv43UNet


class TestWWFCAV43(unittest.TestCase):
    def test_validate_then_retrieve_shape_and_gradients(self):
        module = ValidateThenRetrieve(channels=32, condition_dim=17).train()
        x = torch.randn(2, 32, 8, 8, requires_grad=True)
        bands = [torch.randn_like(x) for _ in range(3)]
        condition = torch.randn(2, 17)
        output = module(x, *bands, condition)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertIsNotNone(module.validate.to_q.weight.grad)
        self.assertIsNotNone(module.retrieve.to_q.weight.grad)
        self.assertIsNotNone(module.validation_gate[-1].weight.grad)
        self.assertIsNotNone(module.retrieval_scale_logit.grad)

    def test_v43_unet_shape_and_diagnostics(self):
        model = WWFCAv43UNet(widths=(32, 32, 32, 32), layers=1).eval()
        sample = torch.randn(1, 2, 32, 32)
        with torch.no_grad():
            output = model(sample, torch.tensor([500]), sigma=torch.tensor([0.2])).sample
        diagnostics = model.diagnostics()
        self.assertEqual(output.shape, (1, 1, 32, 32))
        self.assertTrue(0 < diagnostics["gamma"] < 0.1)
        self.assertTrue(0 <= diagnostics["validation_entropy"] <= 1.1)
        self.assertTrue(0 <= diagnostics["retrieval_entropy"] <= 1.1)
        self.assertTrue(0 < diagnostics["retrieval_scale"] < 0.5)


if __name__ == "__main__":
    unittest.main()
