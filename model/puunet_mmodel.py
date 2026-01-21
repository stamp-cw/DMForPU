import torch

from model.model_setup import ModelSetup
from selector.mmodel_selector import register_mmodel

@register_mmodel(name='PUUnetMModel')
class UnetMModel:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        model_setup = ModelSetup(self.config, config.logger)
        self.model = model_setup.model

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def train_predict(self, gt):
        self.gt = gt.to(self.device)
        self.pred = self.model(self.gt)
        return self.pred

    def eval_predict(self, gt):
        with torch.no_grad():
            self.gt = gt.to(self.device)
            self.pred = self.model(self.gt)
        return self.pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()