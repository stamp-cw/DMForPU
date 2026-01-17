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
        if self.config.diffusion.on_conditioning and self.config.diffusion.use_controlnet:
            self.controlnet_model = model_setup.controlnet_model

    def _create_diffusion(self):
        if self.config.diffusion.use_custom_model:
            self._setup_model()
            self.diffusion = _DIFFUSIONS[self.config.diffusion.name](self.config, self.model, self.controlnet_model)
        else:
            self.diffusion = _DIFFUSIONS[self.config.diffusion.name](self.config)