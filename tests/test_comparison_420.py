import unittest
import torch
from diffusers import UNet2DConditionModel
from experiments.compare_420 import ComparisonModel


class Comparison420Tests(unittest.TestCase):
    def test_matched_hf_and_dcc_have_same_architecture_except_condition_channels(self):
        for variant,channels in [('hf_matched',2),('dcc_only',3)]:
            model=ComparisonModel(variant,widths=(32,32,32,32),layers=1,cross_dim=96)
            self.assertIs(type(model.backbone),UNet2DConditionModel)
            self.assertEqual(model.backbone.config.in_channels,channels)
            self.assertEqual(model.backbone.config.block_out_channels,(32,32,32,32))

    def test_all_new_paths_have_finite_trainable_outputs(self):
        torch.manual_seed(5);w=torch.randn(2,1,32,32);x=torch.randn_like(w)
        for variant in ('hf_matched','dcc_only','wwfca_only','fdu','dlpu'):
            model=ComparisonModel(variant,widths=(32,32,32,32),layers=1,cross_dim=96)
            norm,phi,_=model(x,w,torch.tensor([10,20]),torch.zeros(2))
            self.assertEqual(phi.shape,w.shape)
            norm.square().mean().backward()
            self.assertTrue(torch.isfinite(phi).all())
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if variant=='dlpu':
                model.eval();a=model.sample(w,torch.zeros(2),torch.Generator().manual_seed(1))
                b=model.sample(w,torch.ones(2),torch.Generator().manual_seed(2))
                torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
