from model.model_setup import ModelSetup
from selector.data_selector import _DATA_LOADERS

class ModelTester:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = config.test.device
        self.model = ModelSetup(self.config, self.logger).vae
        self.data_loader = _DATA_LOADERS(self.config)
        self.test_iter = iter(self.eval_loader)