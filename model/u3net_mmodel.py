# import torch
#
# from model.model_setup import ModelSetup
# from selector.mmodel_selector import register_mmodel
#
# from scipy.stats import norm
# import numpy as np
#
# @register_mmodel(name='U3NetMModel')
# class U3NetMModel:
#     def __init__(self, config):
#         self.config = config
#         self.device = config.training.device
#         model_setup = ModelSetup(self.config, config.logger)
#         self.model = model_setup.model
#
#     def get_Gaussian_Noise(self, h, w, SNR):
#         reqSNR = 10 ** (SNR / 10)
#         sigPower = 1
#         sigPower = 10 ** (sigPower / 10)
#         noisePower = sigPower / reqSNR
#         std = np.sqrt(noisePower)
#         noise = std * norm.rvs(0, 1, size=(h, w)).astype(np.float32)
#         return noise, std
#
#     def Wrap(self, x):
#         return np.remainder(x+torch.pi, np.ones_like(x) * (2*torch.pi))-torch.pi
#
#     def grad_op(self, x):
#         res = np.zeros((*x.shape, 2))
#         res[..., :, 1:, 0] = x[..., :, 1:] - x[..., :, :-1]
#         res[..., 1:, :, 1] = x[..., 1:, :] - x[..., :-1, :]
#         return res.astype(np.float32)
#
#     def setup_data(self,batch):
#         self.wrapped = batch['wrapped'].to(self.device)
#         self.gt = batch['unwrapped'].to(self.device)
#         self.pred_batch = batch
#         wrapped_numpy = self.wrapped.cpu().numpy()
#         # snr = self.config.training.snr
#         snr = 0
#         noise, std = self.get_Gaussian_Noise(wrapped_numpy.shape[-2], wrapped_numpy.shape[-1], snr)
#         noise = self.Wrap(wrapped_numpy+noise)-wrapped_numpy
#         WGy = self.Wrap(self.grad_op(wrapped_numpy))
#         WGy_plus, WGy_minus = self.Wrap(self.grad_op(wrapped_numpy+noise)), self.grad_op(wrapped_numpy-noise)
#         self.WGy = torch.from_numpy(WGy).to(self.device)
#         self.WGy_plus = torch.from_numpy(WGy_plus).to(self.device)
#         self.WGy_minus = torch.from_numpy(WGy_minus).to(self.device)
#         self.std = torch.tensor(std).to(self.device)
#
#     def setup_train(self):
#         self.model.train()
#
#     def setup_eval(self):
#         self.model.eval()
#
#     def train_predict(self, batch):
#         # pred, self.pred_list = self.model(self.wrapped)
#         WGy_plus = self.WGy_plus
#         cond = self.std
#         # x_init = torch.ones(WGy_plus[0],1,256,256).to(self.device)
#         x_init = torch.ones(*WGy_plus.shape).to(self.device)
#         a_init = torch.zeros(*x_init.shape, 2).to(self.device)
#         pred, self.pred_list = self.model(WGy_plus, cond, x_init, a_init)
#         self.pred_batch['gt'] = self.gt
#         self.pred_batch['pred'] = pred
#
#
#     def eval_predict(self, batch):
#         self.pred_batch['gt'] = self.gt
#         with torch.no_grad():
#             # x_init = torch.ones(self.WGy.shape[0], 1, self.WGy.shape[-3], self.WGy.shape[-2]).to(self.device)
#             x_init = torch.ones(*self.WGy.shape).to(self.device)
#             a_init = torch.zeros(*x_init.shape, 2).to(self.device)
#             # pred = self.model(self.wrapped)
#             cond = self.std
#             pred = self.model(self.WGy, cond, x_init, a_init)
#             self.pred_batch['pred'] = pred
#
#     @property
#     def optimize_parameters(self):
#         return self.model.parameters()

import torch
import numpy as np
from scipy.stats import norm

from model.model_setup import ModelSetup
from selector.mmodel_selector import register_mmodel


@register_mmodel(name='U3NetMModel')
class U3NetMModel:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        model_setup = ModelSetup(self.config, config.logger)
        self.model = model_setup.model

    def get_Gaussian_Noise(self, h, w, SNR):
        reqSNR = 10 ** (SNR / 10)
        sigPower = 10 ** (1 / 10)
        noisePower = sigPower / reqSNR
        std = np.sqrt(noisePower)
        noise = std * norm.rvs(0, 1, size=(h, w)).astype(np.float32)
        return noise, std

    def Wrap(self, x):
        # x: numpy array
        return np.remainder(x + np.pi, 2 * np.pi) - np.pi

    def grad_op(self, x):
        """
        x: numpy array, shape (B, C, H, W)
        return: (B, C, H, W, 2)
        """
        B, C, H, W = x.shape
        res = np.zeros((B, C, H, W, 2), dtype=np.float32)

        # d/dy (height)
        res[:, :, 1:, :, 0] = x[:, :, 1:, :] - x[:, :, :-1, :]

        # d/dx (width)
        res[:, :, :, 1:, 1] = x[:, :, :, 1:] - x[:, :, :, :-1]

        return res

    def setup_data(self, batch):
        self.wrapped = batch['wrapped'].to(self.device)
        self.gt = batch['unwrapped'].to(self.device)
        self.pred_batch = batch

        wrapped_numpy = self.wrapped.cpu().numpy()
        assert wrapped_numpy.ndim == 4, wrapped_numpy.shape  # (B,C,H,W)

        snr = 0
        noise, std = self.get_Gaussian_Noise(
            wrapped_numpy.shape[-2],
            wrapped_numpy.shape[-1],
            snr
        )

        # broadcast to (B,C,H,W)
        noise = self.Wrap(wrapped_numpy + noise) - wrapped_numpy

        WGy = self.Wrap(self.grad_op(wrapped_numpy))
        WGy_plus = self.Wrap(self.grad_op(wrapped_numpy + noise))
        WGy_minus = self.grad_op(wrapped_numpy - noise)

        self.WGy = torch.from_numpy(WGy).to(self.device)
        self.WGy_plus = torch.from_numpy(WGy_plus).to(self.device)
        self.WGy_minus = torch.from_numpy(WGy_minus).to(self.device)
        # self.std = torch.tensor(std, device=self.device)
        # B = self.wrapped.shape[0]
        self.std = torch.full((self.wrapped.shape[0], 1), std, device=self.device)


    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def train_predict(self, batch):
        WGy_plus = self.WGy_plus
        cond = self.std

        # x_init: same shape as WGy_plus
        # x_init = torch.ones_like(WGy_plus)
        x_init = torch.ones(*WGy_plus.shape[:-1], device=self.device)
        a_init = torch.zeros(*x_init.shape, 2, device=self.device)

        pred, self.pred_list = self.model(WGy_plus, cond, x_init, a_init)

        self.pred_batch['gt'] = self.gt
        self.pred_batch['pred'] = pred

    def eval_predict(self, batch):
        self.pred_batch['gt'] = self.gt
        cond = self.std

        with torch.no_grad():
            # x_init = torch.ones_like(self.WGy)
            x_init = torch.ones(self.WGy.shape[:-1], device=self.device)
            a_init = torch.zeros(*x_init.shape, 2, device=self.device)
            pred, _ = self.model(self.WGy, cond, x_init, a_init)
            self.pred_batch['pred'] = pred

    @property
    def optimize_parameters(self):
        return self.model.parameters()
