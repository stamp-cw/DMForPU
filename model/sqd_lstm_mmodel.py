import torch

from model.model_setup import ModelSetup
from selector.mmodel_selector import register_mmodel

@register_mmodel(name='SqdLstmMModel')
class SqdLstmMModel:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        model_setup = ModelSetup(self.config, config.logger)
        self.model = model_setup.model

    def setup_data(self,batch):
        self.gt = batch['unwrapped'].to(self.device)
        self.pred_batch = batch

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def train_predict(self, batch):
        self.pred_batch['gt'] = self.gt
        self.pred_batch['pred'] = self.model(self.gt)

    def eval_predict(self, batch):
        self.pred_batch['gt'] = self.gt
        with torch.no_grad():
            self.pred_batch['pred'] = self.model(self.gt)

    @property
    def optimize_parameters(self):
        return self.model.parameters()