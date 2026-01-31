from functools import cached_property

import torch
import tqdm

from model.mmodel_setup import MModelSetup
from meter.meter_setup import MeterSetup
from selector.data_selector import _DATA_LOADERS

class ModelTester:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.epoch = config.sampling_from_epoch
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.io.tensorboard_path)
            config.writer = self.writer
        self.meter = MeterSetup(self.config, self.logger).meter
        self.device = config.val.device
        self.data_loader = _DATA_LOADERS(self.config)
        self.mmodel = MModelSetup(self.config, self.logger).mmodel

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.val_ckpt_file_path}")
        state_dict = torch.load(self.config.io.val_ckpt_file_path, map_location=self.device, weights_only=True)['model']
        is_multi_card = any(k.startswith("module.") for k in state_dict.keys())
        if is_multi_card:
            self.logger.info("Detected multi-card checkpoint. Stripping 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            self.logger.info("Detected single-card checkpoint.")
        self.mmodel.model.load_state_dict(state_dict)


    def test(self):
        self._sample()
        self._val(self.epoch)

    def sample(self):
        self._sample()

    def val(self, epoch):
        self._val(epoch)

    def _sample(self):
        from run.sample_model import ModelSampler as Sampler
        self.config.sampling.batch_size = self.config.test.batch_size
        self.config.sampling.total_samples = self.config.test.total_samples
        sampler = Sampler(self.config)
        sampler.sampling_loader = self.sampling_loader
        sampler.mmodel = self.mmodel
        sampler.sample()

    def _val(self, epoch):
        from run.val_model import ModelValidator as Valuator
        self.config.val.batch_size = self.config.test.batch_size
        valuator = Valuator(self.config)
        valuator.epoch = epoch
        valuator.meter = self.meter
        valuator.meter.writer = self.writer
        valuator.meter.mode = 'test'
        valuator.diffusion = self.mmodel
        valuator.val_loader = self.val_loader
        valuator.valuate()

    @property
    def val_loader(self):
        return self.data_loader.test_loader

    @property
    def sampling_loader(self):
        return self.data_loader.test_loader