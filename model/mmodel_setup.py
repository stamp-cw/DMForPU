from selector.mmodel_selector import _MMODELS
import torch

class MModelSetup:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_mmodel()
        self._log_model_info()

    def _create_mmodel(self):
        self.mmodel = _MMODELS[self.config.mmodel.name](self.config)
        # self.model = torch.nn.DataParallel(self.model)

    def _log_model_info(self):
        self.logger.debug(self.mmodel)