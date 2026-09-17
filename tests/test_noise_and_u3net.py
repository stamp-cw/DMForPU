import math
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dataset.SyntheticPUMatNoise import SyntheticPUMatNoise
from model.u3net_mmodel import U3NetMModel
from run.losses import U3NetLossType
from run.test_model import ModelTester


class NoiseRegressionTest(unittest.TestCase):
    def test_noise_matches_upstream_formula(self):
        dataset = SyntheticPUMatNoise.__new__(SyntheticPUMatNoise)
        wrapped = torch.linspace(-math.pi, math.pi, 4096).reshape(1, 64, 64)

        torch.manual_seed(42)
        actual = dataset.get_Gaussian_Noise(wrapped, 10)

        upstream_std = math.sqrt(10 ** (1 / 10) / 10 ** (10 / 10))
        torch.manual_seed(42)
        raw_noise = upstream_std * torch.randn_like(wrapped)
        expected = dataset.wrap_phase(wrapped + raw_noise) - wrapped

        torch.testing.assert_close(actual, expected)
        circular_noise = dataset.wrap_phase(actual)
        self.assertLess(abs(float(circular_noise.mean())), 0.02)
        self.assertAlmostEqual(float(circular_noise.std()), upstream_std, delta=0.02)


class U3NetRegressionTest(unittest.TestCase):
    def _wrapper_without_model(self):
        wrapper = U3NetMModel.__new__(U3NetMModel)
        wrapper.device = "cpu"
        wrapper.config = SimpleNamespace(data=SimpleNamespace(noise_snr=10))
        return wrapper

    def test_setup_data_defines_std_and_uses_original_gradient_order(self):
        wrapper = self._wrapper_without_model()
        wrapped = torch.arange(24, dtype=torch.float32).reshape(2, 1, 3, 4)
        batch = {"wrapped": wrapped, "unwrapped": wrapped.clone()}

        np.random.seed(42)
        wrapper.setup_data(batch)

        expected_std = math.sqrt(10 ** (1 / 10) / 10 ** (10 / 10))
        self.assertEqual(tuple(wrapper.std.shape), (2, 1))
        self.assertTrue(torch.allclose(wrapper.std, torch.full((2, 1), expected_std)))
        self.assertEqual(tuple(wrapper.WGy.shape), (2, 1, 3, 4, 2))
        gradient = wrapper.grad_op(wrapped.numpy())
        torch.testing.assert_close(wrapper.WGy[..., 0], torch.from_numpy(
            wrapper.Wrap(gradient)[..., 0]
        ))
        self.assertTrue(np.all(gradient[:, :, :, 1:, 0] == 1))
        self.assertTrue(np.all(gradient[:, :, 1:, :, 1] == 4))

    def test_loss_is_original_self_supervised_gradient_loss(self):
        loss = U3NetLossType(SimpleNamespace(
            loss_type=SimpleNamespace(name="U3NetLoss"),
            train_meter=object(),
        ))
        predictions = [
            torch.zeros(1, 1, 4, 4, requires_grad=True),
            torch.ones(1, 1, 4, 4, requires_grad=True),
        ]
        wrapper = SimpleNamespace(
            pred_list=predictions,
            WGy_minus=torch.zeros(1, 1, 4, 4, 2),
        )

        actual = loss(wrapper)
        expected = sum(
            loss.Loss_SR(wrapper.WGy_minus, prediction) / (2 - j)
            for j, prediction in enumerate(predictions)
        )

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(all(prediction.grad is not None for prediction in predictions))

    def test_distillation_phase_freezes_and_restores_teacher(self):
        wrapper = self._wrapper_without_model()
        wrapper.config.training = SimpleNamespace(
            brand_new_epochs=700,
            distill_epochs=200,
        )
        wrapper.model = torch.nn.Conv2d(1, 1, 1)
        wrapper.teacher_model = None
        wrapper.is_distilling = False

        self.assertFalse(wrapper.configure_training_phase(499))
        self.assertFalse(wrapper.is_distilling)
        self.assertTrue(wrapper.configure_training_phase(500))
        self.assertTrue(wrapper.is_distilling)
        self.assertFalse(wrapper.teacher_model.training)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in wrapper.teacher_model.parameters()
        ))

        teacher_before = copy.deepcopy(wrapper.teacher_model.state_dict())
        with torch.no_grad():
            wrapper.model.weight.add_(1)
        for name, value in wrapper.teacher_model.state_dict().items():
            torch.testing.assert_close(value, teacher_before[name])

        state = wrapper.training_state_dict()
        restored = self._wrapper_without_model()
        restored.model = torch.nn.Conv2d(1, 1, 1)
        restored.teacher_model = None
        restored.is_distilling = False
        restored.load_training_state_dict(state)
        self.assertTrue(restored.is_distilling)
        for name, value in restored.teacher_model.state_dict().items():
            torch.testing.assert_close(value, state["teacher_model"][name])

    def test_distillation_uses_upstream_gradient_loss(self):
        loss = U3NetLossType(SimpleNamespace(
            loss_type=SimpleNamespace(name="U3NetLoss"),
            train_meter=object(),
        ))
        prediction = torch.randn(1, 1, 4, 4, requires_grad=True)
        wrapper = SimpleNamespace(
            is_distilling=True,
            pred_list=[prediction],
            distill_target=torch.randn(1, 1, 4, 4),
        )

        actual = loss(wrapper)
        expected = loss.Loss_SD(wrapper.distill_target, prediction)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(prediction.grad)

    def test_model_tester_passes_mmodel_to_validator(self):
        created = []

        class FakeValidator:
            def __init__(self, config):
                self.config = config
                created.append(self)

            def valuate(self):
                self.did_valuate = True

        tester = ModelTester.__new__(ModelTester)
        tester.config = SimpleNamespace(val=SimpleNamespace(batch_size=1), test=SimpleNamespace(batch_size=1))
        tester.epoch = 3
        tester.meter = SimpleNamespace(writer=None, mode=None)
        tester.writer = None
        tester.mmodel = object()
        tester.data_loader = SimpleNamespace(test_loader=[object()])

        with patch("run.val_model.ModelValidator", FakeValidator):
            tester._val(tester.epoch)

        self.assertIs(created[0].mmodel, tester.mmodel)
        self.assertTrue(created[0].did_valuate)


if __name__ == "__main__":
    unittest.main()
