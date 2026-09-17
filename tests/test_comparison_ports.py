from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from model.lstm.sqd_lstm import JointConvSQDLSTMNet
from model.transformer.uformer import Uformer
from model.unet.punet import PUNet


ROOT = Path(__file__).resolve().parents[1]


class ComparisonPortRegressionTest(unittest.TestCase):
    def test_punet_matches_official_parameter_count_and_shape(self):
        model = PUNet()

        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 2_147_393)
        self.assertEqual(len(model.dilated), 8)
        self.assertEqual(len(model.residual), 10)
        self.assertEqual(model(torch.randn(2, 1, 32, 32)).shape, (2, 1, 32, 32))

    def test_sqd_vertical_sequence_matches_keras_spatial_permute(self):
        feature = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

        actual = JointConvSQDLSTMNet._to_vertical_sequence(feature)
        expected = feature.permute(0, 3, 2, 1).reshape(2, 5 * 4, 3)

        torch.testing.assert_close(actual, expected)

    def test_sqd_vertical_sequence_round_trip(self):
        feature = torch.randn(2, 3, 4, 5)
        sequence = JointConvSQDLSTMNet._to_vertical_sequence(feature)

        restored = JointConvSQDLSTMNet._from_vertical_sequence(sequence, 4, 5)

        torch.testing.assert_close(restored, feature)

    def test_sqd_batch_norm_matches_keras_defaults(self):
        model = JointConvSQDLSTMNet(SimpleNamespace())

        self.assertEqual(model.c1.bn.eps, 1e-3)
        self.assertEqual(model.c1.bn.momentum, 0.01)

    def test_all_sqd_configs_select_official_loss_and_meter(self):
        config_paths = sorted((ROOT / "configs").glob("sqd_lstm*.yaml"))

        self.assertTrue(config_paths)
        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                config_text = config_path.read_text(encoding="utf-8")
                self.assertIn("name: SqdLstmLoss", config_text)
                self.assertIn("name: SqdLstmMeter", config_text)
                self.assertNotIn("name: DLPULoss", config_text)
                self.assertNotIn("name: PUNetMeter", config_text)

    def test_uformer_uses_configured_sample_size(self):
        config = SimpleNamespace(model=SimpleNamespace(sample_size=32))
        model = Uformer(
            config,
            img_size=256,
            embed_dim=8,
            depths=[1] * 9,
            num_heads=[1] * 9,
            win_size=4,
        )

        self.assertEqual(model.reso, 32)
        self.assertEqual(model.encoderlayer_0.input_resolution, (32, 32))
