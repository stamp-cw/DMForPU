import unittest
import torch
from diffusers import UNet2DConditionModel
from diffusion.chen_hf_diffusion import (ChenHFConfig,ChenHFDiffusion,GradientUnroll,
    gradient,adjoint,wrap,paired_recorruption,sr_loss,sd_loss)


class ChenHFTests(unittest.TestCase):
    def test_gradient_adjoint_and_boundary(self):
        torch.manual_seed(1);x=torch.randn(2,1,9,11,dtype=torch.float64);g=torch.randn(*x.shape,2,dtype=x.dtype)
        torch.testing.assert_close((gradient(x)*g).sum(),(x*adjoint(g)).sum())
        torch.testing.assert_close(adjoint(g).sum((-2,-1)),torch.zeros(2,1,dtype=x.dtype),atol=1e-12,rtol=0)

    def test_paired_noise_and_circular_loss(self):
        w=torch.randn(2,1,8,8);s=torch.tensor([0.,.5]);z=torch.randn_like(w)
        plus,minus=paired_recorruption(w,s,z)
        torch.testing.assert_close(minus,gradient(2*w-plus))
        p=w+2*torch.pi;self.assertLess(float(sr_loss(p,gradient(w))),1e-10)

    def test_physics_preserves_consistent_phase_and_constant_offset(self):
        x=torch.linspace(0,2,64).reshape(1,1,8,8);net=GradientUnroll(ChenHFConfig())
        p,_=net(x,wrap(x),torch.zeros(1));torch.testing.assert_close(p,x)
        p2,_=net(x+4,wrap(x),torch.zeros(1));torch.testing.assert_close(p2,p+4)

    def test_sparse_residual_limits_damage_when_itoh_condition_fails(self):
        x=(torch.arange(12).float()*4)[None,None,None].expand(1,1,12,12)
        robust=GradientUnroll(ChenHFConfig(adaptive=False,sparse=True))
        plain=GradientUnroll(ChenHFConfig(adaptive=False,sparse=False))
        a,_=robust(x,wrap(x),torch.zeros(1));b,_=plain(x,wrap(x),torch.zeros(1))
        self.assertLess(float((a-x).abs().mean().detach()),float((b-x).abs().mean().detach()))

    def test_direct_hf_backbone_training_and_seeded_sampling(self):
        torch.manual_seed(5);cfg=ChenHFConfig(image_size=16,channels=(8,16,16),inner_steps=1)
        model=ChenHFDiffusion(cfg)
        self.assertIs(type(model.unet),UNet2DConditionModel)
        x=torch.randn(2,1,16,16);w=wrap(x);sigma=torch.tensor([0.,.2])
        norm,phi,_=model(x,w,torch.tensor([10,30]),sigma)
        loss=norm.square().mean()+sr_loss(phi,gradient(w));loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.physics.parameters()))
        model.eval()
        a=model.sample(w,sigma,torch.Generator().manual_seed(42),steps=2)
        b=model.sample(w,sigma,torch.Generator().manual_seed(42),steps=2)
        torch.testing.assert_close(a,b)

    def test_distillation_detaches_teacher(self):
        student=torch.randn(1,1,8,8,requires_grad=True);teacher=torch.randn_like(student,requires_grad=True)
        sd_loss(student,teacher).backward();self.assertIsNone(teacher.grad);self.assertIsNotNone(student.grad)


if __name__=='__main__':unittest.main()
