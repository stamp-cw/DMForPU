from diffusion.diffusion_setup import DiffusionSetup
from meter.meter_setup import MeterSetup
from selector.data_selector import _DATA_LOADERS

class Tester:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.epoch = config.sampling_from_epoch
        self.meter = MeterSetup(self.config, self.logger).meter
        self.device = config.val.device
        self.data_loader = _DATA_LOADERS(self.config)
        self.diffusion = DiffusionSetup(self.config, self.logger).diffusion

    def test(self):
        self._sample()
        self._val(self.epoch)

    def sample(self):
        self._sample()

    def val(self, epoch):
        self._val(epoch)

    def _sample(self):
        from run.sample import Sampler
        self.config.sampling.batch_size = self.config.test.batch_size
        self.config.sampling.total_samples = self.config.test.batch_size
        sampler = Sampler(self.config)
        sampler.sampling_loader = self.sampling_loader
        sampler.diffusion = self.diffusion
        sampler.sample()

    def _val(self, epoch):
        from run.val import Valuator
        self.config.val.batch_size = self.config.test.batch_size
        valuator = Valuator(self.config)
        valuator.epoch = epoch
        valuator.meter = self.meter
        valuator.meter.mode = 'test'
        valuator.diffusion = self.diffusion
        valuator.val_loader = self.val_loader
        valuator.valuate()

    @property
    def val_loader(self):
        return self.data_loader.eval_loader

    @property
    def sampling_loader(self):
        return self.data_loader.eval_loader