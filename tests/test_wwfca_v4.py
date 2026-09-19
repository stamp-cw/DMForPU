import unittest

import torch
from diffusers import UNet2DModel

from model.fdunet.wwfca_v4 import GlobalFrequencyCrossAttention, WWFCAv4UNet


def hf(widths=(32,32,32,32)):
    return UNet2DModel(sample_size=32,in_channels=2,out_channels=1,layers_per_block=1,
        block_out_channels=widths,down_block_types=("DownBlock2D",)*4,
        up_block_types=("UpBlock2D",)*4,add_attention=False)


class TestWWFCAV4(unittest.TestCase):
    def test_global_attention_shape_and_gradients(self):
        module=GlobalFrequencyCrossAttention(32,heads=4,max_size=16,dropout=0.0)
        bands=[torch.randn(2,32,8,8,requires_grad=True) for _ in range(4)]
        output=module(*bands,condition=torch.randn(2,129))
        self.assertEqual(output.shape,bands[0].shape)
        output.square().mean().backward()
        self.assertGreater(float(module.to_q.weight.grad.abs().sum()),0.0)
        self.assertGreater(float(module.high_in.weight.grad.abs().sum()),0.0)

    def test_off_variant_exactly_preserves_hf_backbone(self):
        torch.manual_seed(9);source=hf().eval()
        target=WWFCAv4UNet(False,widths=(32,32,32,32),layers=1).eval()
        target.load_hf_backbone(source.state_dict())
        x=torch.randn(2,2,32,32);t=torch.tensor([10,500]);sigma=torch.tensor([0.0,1.0])
        with torch.no_grad():
            expected=source(x,t).sample;actual=target(x,t,sigma).sample
        self.assertTrue(torch.equal(expected,actual))

    def test_enabled_branch_is_small_and_trainable(self):
        source=hf().eval();model=WWFCAv4UNet(True,widths=(32,32,32,32),layers=1)
        model.load_hf_backbone(source.state_dict())
        x=torch.randn(2,2,32,32);t=torch.tensor([20,700]);sigma=torch.tensor([0.0,0.5])
        y=model(x,t,sigma).sample;loss=y.square().mean()+model.energy_penalty();loss.backward()
        self.assertGreater(float(model.adapter.cross.to_q.weight.grad.abs().sum()),0.0)
        self.assertGreater(float(model.adapter.gamma_logit.grad.abs()),0.0)
        diagnostics=model.diagnostics()
        self.assertAlmostEqual(diagnostics["gamma"],0.02,places=5)
        self.assertLess(diagnostics["residual_rms_ratio"],0.01)


if __name__=="__main__":unittest.main()
