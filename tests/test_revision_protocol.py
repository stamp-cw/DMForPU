import unittest
import numpy as np
import torch
from types import SimpleNamespace as NS
from experiments.revision_study import observed, metric_row, config_for


class RevisionProtocolTests(unittest.TestCase):
    def test_restormer_microbatch_accumulation_preserves_gradient(self):
        from model.transformer.restormer import Restormer
        torch.manual_seed(12)
        model=Restormer(NS(),dim=8,num_blocks=[1,1,1,1],num_refinement_blocks=1)
        x=torch.randn(3,1,16,16);y=torch.randn_like(x)
        torch.nn.functional.l1_loss(model(x),y).backward()
        full=[p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for i in range(3):
            (torch.nn.functional.l1_loss(model(x[i:i+1]),y[i:i+1])/3).backward()
        for p,g in zip(model.parameters(),full):
            torch.testing.assert_close(p.grad,g,atol=2e-6,rtol=2e-4)

    def test_u3_receives_per_image_noise_condition(self):
        from model.u3net_mmodel import U3NetMModel
        wrapper=U3NetMModel.__new__(U3NetMModel)
        wrapper.config=NS(data=NS(noise_snr=30));wrapper.device='cpu'
        wrapper.setup_data({'wrapped':torch.zeros(2,1,8,8),'unwrapped':torch.zeros(2,1,8,8),
                            'snr_db':torch.tensor([0.,20.])})
        self.assertAlmostEqual(float(wrapper.std[0]/wrapper.std[1]),10.,places=5)

    def test_clean_observation_is_exact_and_noise_is_reproducible(self):
        wrapped=torch.linspace(-3,3,32).reshape(2,1,4,4)
        snr=torch.tensor([1000,10])
        a=observed(wrapped,snr,torch.Generator().manual_seed(42))
        b=observed(wrapped,snr,torch.Generator().manual_seed(42))
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a[0],wrapped[0],atol=0,rtol=0)
        self.assertFalse(torch.equal(a[1],wrapped[1]))

    def test_integer_reference_alignment_does_not_hide_local_errors(self):
        target=np.arange(16,dtype=float).reshape(4,4)/4
        wrapped=np.angle(np.exp(1j*target))
        result=metric_row(target+2*np.pi,target,wrapped)
        self.assertAlmostEqual(result['aligned_mae'],0,places=10)
        self.assertAlmostEqual(result['mae'],2*np.pi)
        distorted=target+2*np.pi;distorted[0,0]+=1
        self.assertGreater(metric_row(distorted,target,wrapped)['aligned_mae'],0)

    def test_ablation_flags_and_short_u3_budget_are_explicit(self):
        for variant,channels,name in [('base',1,'FDUNetV1'),('dcc',2,'FDUNetV1'),('wwf',1,'FDUNet'),('both',2,'FDUNet')]:
            cfg=config_for('synthetic','fdu',30,42,variant)
            self.assertEqual(cfg['diffusion']['conditioning_channels'],channels)
            self.assertEqual(cfg['model']['name'],name)
        cfg=config_for('synthetic','u3net',70,42)
        self.assertEqual(cfg['training']['distill_start_epoch'],50)
        self.assertEqual(cfg['training']['distill_epochs'],20)


if __name__=='__main__':unittest.main()
