from selector.model_selector import _MODELS
import torch

class ModelSetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_model()
        self._set_mode()
        self._log_model_info()

    def _create_model(self):
        self.model = _MODELS[self.config.model.name](self.config).to(self.config.training.device)
        self.model = torch.nn.DataParallel(self.model)
        # if self.config.diffusion.use_controlnet:
        #     self.controlnet_model = _MODELS[self.config.controlnet_model.name](self.config).to(self.config.training.device)
        #     self.controlnet_model = torch.nn.DataParallel(self.controlnet_model)

    def _log_model_info(self):
        self.logger.debug(self.model)
        # if self.config.diffusion.use_controlnet:
        #     self.logger.debug(self.controlnet_model)

    def _set_mode(self):
        if self.config.mode == 'train':
            self.model.train()
            if self.config.diffusion.use_controlnet:
                self.controlnet_model.train()
        elif self.config.mode == 'sample':
            self.model.eval()
            if self.config.diffusion.use_controlnet:
                self.controlnet_model.eval()