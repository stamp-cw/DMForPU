import copy
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import yaml

from model.lstm.sqd_lstm import JointConvSQDLSTMNet, KerasSameConvTranspose2d
from model.optimizer import OptimizerFN
from model.transformer.uformer import Uformer
from model.u3net_mmodel import U3NetMModel
from run.train_model import EpochFN
from utils.metrics import nrmse_metric
from utils.util import AverageMeter


class PortFixTests(unittest.TestCase):
    def test_amp_unscales_before_clipping_and_skips_nonfinite_update(self):
        config = NS(optim=NS(warmup=0, grad_clip=1, lr=0.1))
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = torch.amp.GradScaler('cpu')
        scaler.scale(parameter * 0.001).backward()
        OptimizerFN(config)(optimizer, [parameter], epoch=0, scaler=scaler)
        self.assertAlmostEqual(parameter.item(), 0.9999, places=6)
        optimizer.zero_grad()
        before, scale = parameter.detach().clone(), scaler.get_scale()
        scaler.scale(parameter * float('inf')).backward()
        OptimizerFN(config)(optimizer, [parameter], epoch=1, scaler=scaler)
        torch.testing.assert_close(parameter, before)
        self.assertLess(scaler.get_scale(), scale)

    def test_sqd_transpose_matches_tensorflow_impulse(self):
        layer = KerasSameConvTranspose2d(1, 1)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(1, 10).reshape(1, 1, 3, 3))
            layer.bias.zero_()
        x = torch.zeros(1, 1, 2, 2, requires_grad=True)
        with torch.no_grad():
            x[0, 0, 0, 0] = 1
        expected = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0.]])
        actual = layer(x)
        torch.testing.assert_close(actual.squeeze(), expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_sqd_initialization_and_shape(self):
        model = JointConvSQDLSTMNet(None)
        for lstm in (model.lstm_h, model.lstm_v):
            for suffix in ('', '_reverse'):
                bias = getattr(lstm, 'bias_ih_l0' + suffix) + getattr(lstm, 'bias_hh_l0' + suffix)
                expected = torch.zeros(128); expected[32:64] = 1
                torch.testing.assert_close(bias, expected)
                recurrent = getattr(lstm, 'weight_hh_l0' + suffix)
                torch.testing.assert_close(recurrent.T @ recurrent, torch.eye(32), atol=1e-6, rtol=1e-5)
                self.assertFalse(getattr(lstm, 'bias_hh_l0' + suffix).requires_grad)
        with torch.no_grad():
            self.assertEqual(model(torch.randn(2, 1, 32, 48)).shape, (2, 1, 32, 48))

    def test_uformer_config_controls_modules_and_residual(self):
        cfg = NS(model=NS(sample_size=32, embed_dim=8, depths=[1]*9,
                          num_heads=[1]*9, win_size=4, modulator=True, residual=True))
        model = Uformer(cfg).eval()
        self.assertTrue(any(getattr(m, 'modulator', None) is not None for m in model.modules()))
        with torch.no_grad():
            for p in model.output_proj.parameters():
                p.zero_()
            x = torch.randn(1, 1, 32, 32)
            torch.testing.assert_close(model(x), x)
            model.residual = False
            torch.testing.assert_close(model(x), torch.zeros_like(x))
        cfg.model.modulator = False
        legacy = Uformer(cfg)
        self.assertFalse(any(getattr(m, 'modulator', None) is not None for m in legacy.modules()))

    def test_shipped_u3net_configs_schedule_both_phases(self):
        for path in (Path(__file__).resolve().parents[1] / 'configs').glob('u3net*.yaml'):
            cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
            self.assertEqual(cfg['training']['distill_start_epoch'], 500)
            self.assertEqual(cfg['training']['brand_new_epochs'], 700)
            self.assertEqual(cfg['training']['distill_epochs'], 200)

    def test_nrmse_is_mean_of_per_image_values(self):
        gt = torch.tensor([[[[0., 1.]]], [[[0., 100.]]]])
        pred = gt + 1
        joint = nrmse_metric(pred, gt)
        separate = (nrmse_metric(pred[:1], gt[:1]) + nrmse_metric(pred[1:], gt[1:])) / 2
        torch.testing.assert_close(joint, separate)
        self.assertAlmostEqual(joint.item(), 0.505, places=6)

    def test_epoch_average_weights_short_batches_and_drops_graphs(self):
        meter = AverageMeter()
        meter.update({'NRMSE': torch.tensor(1., requires_grad=True)}, n=2)
        meter.update({'NRMSE': torch.tensor(4., requires_grad=True)}, n=1)
        actual = meter.avg()['NRMSE']
        self.assertEqual(actual.item(), 2.)
        self.assertFalse(actual.requires_grad)

    def test_distillation_restarts_decay_and_preserves_teacher_on_resume(self):
        wrapper = U3NetMModel.__new__(U3NetMModel)
        wrapper.config = NS(training=NS(brand_new_epochs=700, distill_epochs=200))
        wrapper.model = torch.nn.Linear(1, 1)
        wrapper.teacher_model = None; wrapper.is_distilling = False
        self.assertFalse(wrapper.configure_training_phase(499))
        self.assertTrue(wrapper.configure_training_phase(500))
        self.assertEqual(wrapper.optimization_epoch(500), 0)
        saved = copy.deepcopy(wrapper.training_state_dict())
        with torch.no_grad(): wrapper.model.weight.add_(3)
        wrapper.load_training_state_dict(saved)
        self.assertFalse(wrapper.configure_training_phase(501))
        torch.testing.assert_close(wrapper.teacher_model.weight, saved['teacher_model']['weight'])
        config = NS(optim=NS(lr=0.001, scheduler='exponential', gamma=.99, warmup=0, grad_clip=-1))
        optimizer = torch.optim.SGD(wrapper.model.parameters(), lr=.001)
        OptimizerFN(config)(optimizer, wrapper.model.parameters(), wrapper.optimization_epoch(501))
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .00099)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_epoch_autocasts_forward_and_keeps_scaler(self):
        class Wrapper:
            def __init__(self): self.model = torch.nn.Conv2d(1, 1, 1).cuda()
            @property
            def optimize_parameters(self): return self.model.parameters()
            def setup_data(self, batch): self.batch = batch
            def train_predict(self, batch):
                self.amp_seen = torch.is_autocast_enabled('cuda')
                self.pred_batch = {'pred': self.model(batch), 'gt': torch.zeros_like(batch)}
        class Meter:
            def setup_data(self, batch): self.batch = batch
            def compute_batch_metric(self):
                self.batch_metric_dict = {'L1': (self.batch['pred'].float()-self.batch['gt']).square().mean()}
        meter = Meter()
        cfg = NS(training=NS(device='cuda', amp=True),
                 optim=NS(lr=.01, warmup=0, grad_clip=1), train_meter=meter)
        with patch('run.train_model.LossFN', return_value=lambda model: meter.batch_metric_dict['L1']):
            epoch = EpochFN(OptimizerFN(cfg), cfg)
        model = Wrapper(); optimizer = torch.optim.SGD(model.optimize_parameters, lr=.01)
        scaler = epoch.scaler
        for i in range(2):
            epoch(model, optimizer, meter, i, torch.ones(2, 1, 4, 4, device='cuda'))
        self.assertTrue(model.amp_seen)
        self.assertIs(epoch.scaler, scaler)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.model.parameters()))


if __name__ == '__main__':
    unittest.main()
