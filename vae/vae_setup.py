from selector.vae_selector import _VAES
import torch

class VAESetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_vae()
        self._log_model_info()

    def _create_vae(self):
        self.vae = _VAES[self.config.model.name](self.config)
        # self.model = torch.nn.DataParallel(self.model)

    def _log_model_info(self):
        self.logger.debug(self.vae)