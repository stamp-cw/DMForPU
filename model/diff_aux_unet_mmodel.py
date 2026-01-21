import torch

from model.model_setup import ModelSetup
from selector.mmodel_selector import register_mmodel

@register_mmodel(name='DiffAuxUNetMModel')
class DiffAuxUNetMModel:
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
        t_batch = torch.randint(0, 1, (1,), device=self.config.training.device).long().expand(self.config.training.batch_size)
        self.pred, self.d_feats, self.u_feats = self.model(self.gt, t_batch)
        return self.pred

    def eval_predict(self, gt):
        with torch.no_grad():
            self.gt = gt.to(self.device)
            t_batch = torch.randint(0, 1, (1,), device=self.config.training.device).long().expand(self.config.training.batch_size)
            self.pred, self.d_feats, self.u_feats = self.model(self.gt, t_batch)
        return self.pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()