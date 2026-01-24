from model.model_setup import ModelSetup
from selector.diffusion_selector import _DIFFUSIONS

class DiffusionSetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_diffusion()

    def _setup_model(self):
        model_setup = ModelSetup(self.config, self.logger)
        self.model = model_setup.model

    def _create_diffusion(self):
        self.diffusion = _DIFFUSIONS[self.config.diffusion.name](self.config)