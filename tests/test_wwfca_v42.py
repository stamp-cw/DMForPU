import unittest
import torch
from diffusers import UNet2DModel
from model.fdunet.wwfca_v42 import WWFCAv42UNet


def hf():
    return UNet2DModel(sample_size=32,in_channels=2,out_channels=1,layers_per_block=1,
        block_out_channels=(32,32,32,32),down_block_types=("DownBlock2D",)*4,
        up_block_types=("UpBlock2D",)*4,add_attention=False)


class TestWWFCAV42(unittest.TestCase):
    def test_time_gate_grows_later_and_noise_gate_suppresses_noise(self):
        model=WWFCAv42UNet(True,True,widths=(32,32,32,32),layers=1)
        adapter=model.adapter
        condition=torch.zeros(3,129);progress=torch.tensor([0.0,0.5,1.0]);noise=torch.zeros(3)
        adapter.set_conditions(condition,progress,noise)
        x=torch.randn(3,32,8,8);adapter(x);late=adapter.diagnostics()
        self.assertGreater(late["gamma_max"],late["gamma_min"])
        condition=torch.zeros(2,129);progress=torch.ones(2);noise=torch.tensor([0.0,1.0])
        adapter.set_conditions(condition,progress,noise);adapter(torch.randn(2,32,8,8))
        self.assertGreater(adapter.diagnostics()["gamma_max"],adapter.diagnostics()["gamma_min"])

    def test_gate_parameters_receive_gradients(self):
        model=WWFCAv42UNet(True,True,widths=(32,32,32,32),layers=1)
        y=model(torch.randn(2,2,32,32),torch.tensor([50,900]),torch.tensor([0.0,1.0])).sample
        y.square().mean().backward()
        self.assertGreater(float(model.adapter.time_logit.grad.abs()),0.0)
        self.assertGreater(float(model.adapter.noise_logit.grad.abs()),0.0)

    def test_disabled_path_can_exactly_preserve_hf(self):
        torch.manual_seed(7);source=hf().eval();target=WWFCAv42UNet(False,True,widths=(32,32,32,32),layers=1).eval()
        target.load_hf_backbone(source.state_dict());x=torch.randn(2,2,32,32);t=torch.tensor([0,800]);s=torch.tensor([0.0,1.0])
        with torch.no_grad():self.assertTrue(torch.equal(source(x,t).sample,target(x,t,s).sample))


if __name__=="__main__":unittest.main()
