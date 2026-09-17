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
import copy
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
        self.teacher_model = None
        self.is_distilling = False

    def _student_model(self):
        accelerator = getattr(self.config, 'accelerator', None)
        return accelerator.unwrap_model(self.model) if accelerator is not None else self.model

    @property
    def distill_start_epoch(self):
        distill_epochs = getattr(self.config.training, "distill_epochs", 200)
        return getattr(
            self.config.training,
            "distill_start_epoch",
            max(self.config.training.brand_new_epochs - distill_epochs, 0),
        )

    def optimization_epoch(self, epoch):
        return epoch - self.distill_start_epoch if self.is_distilling else epoch

    def configure_training_phase(self, epoch):
        """Freeze an unwrapped teacher on every rank at the phase boundary."""
        distill_epochs = getattr(self.config.training, 'distill_epochs', 200)
        should_distill = distill_epochs > 0 and epoch >= self.distill_start_epoch
        phase_changed = should_distill and self.teacher_model is None
        if phase_changed:
            self.teacher_model = copy.deepcopy(self._student_model()).eval()
            self.teacher_model.requires_grad_(False)
        self.is_distilling = should_distill
        return phase_changed

    def training_state_dict(self):
        if self.teacher_model is None:
            return {}
        return {
            "teacher_model": self.teacher_model.state_dict(),
            "is_distilling": self.is_distilling,
        }

    def load_training_state_dict(self, state):
        teacher_state = state.get("teacher_model")
        if teacher_state is None:
            return
        self.teacher_model = copy.deepcopy(self._student_model()).eval()
        self.teacher_model.load_state_dict(teacher_state)
        self.teacher_model.requires_grad_(False)
        self.is_distilling = state.get("is_distilling", True)

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

        # Keep the component order used by the original U3Net network:
        # component 0 is the width derivative and component 1 is height.
        res[:, :, :, 1:, 0] = x[:, :, :, 1:] - x[:, :, :, :-1]
        res[:, :, 1:, :, 1] = x[:, :, 1:, :] - x[:, :, :-1, :]

        return res

    def setup_data(self, batch):
        self.wrapped = batch['wrapped'].to(self.device)
        self.gt = batch['unwrapped'].to(self.device)
        self.pred_batch = batch

        wrapped_numpy = self.wrapped.cpu().numpy()
        assert wrapped_numpy.ndim == 4, wrapped_numpy.shape  # (B,C,H,W)

        batch_size, channels, height, width = wrapped_numpy.shape
        snrs = torch.as_tensor(batch.get('snr_db', self.config.data.noise_snr)).reshape(-1)
        if snrs.numel() == 1:
            snrs = snrs.expand(batch_size)
        if snrs.numel() != batch_size:
            raise ValueError('snr_db must contain one value per image')
        noise_and_stds = [
            self.get_Gaussian_Noise(height, width, float(snr))
            for snr in snrs.cpu() for _ in range(channels)
        ]
        noise_samples = [sample[0] for sample in noise_and_stds]
        stds = [noise_and_stds[i * channels][1] for i in range(batch_size)]
        noise = np.stack(noise_samples).reshape(wrapped_numpy.shape)
        noise = self.Wrap(wrapped_numpy + noise) - wrapped_numpy

        WGy = self.Wrap(self.grad_op(wrapped_numpy))
        WGy_plus = self.Wrap(self.grad_op(wrapped_numpy + noise))
        WGy_minus = self.grad_op(wrapped_numpy - noise)

        self.WGy = torch.from_numpy(WGy).to(self.device)
        self.WGy_plus = torch.from_numpy(WGy_plus).to(self.device)
        self.WGy_minus = torch.from_numpy(WGy_minus).to(self.device)
        self.std = torch.tensor(stds, dtype=torch.float32, device=self.device).reshape(batch_size, 1)


    def setup_train(self):
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def setup_eval(self):
        self.model.eval()

    def train_predict(self, batch):
        cond = self.std

        x_init = torch.ones(*self.WGy.shape[:-1], device=self.device)
        a_init = torch.zeros(*x_init.shape, 2, device=self.device)

        if self.is_distilling:
            if self.teacher_model is None:
                raise RuntimeError("U3Net distillation requires a frozen teacher model")
            with torch.no_grad():
                self.distill_target, _ = self.teacher_model(
                    self.WGy_plus, cond, x_init, a_init
                )
            pred, self.pred_list = self.model(self.WGy, cond, x_init, a_init)
        else:
            pred, self.pred_list = self.model(
                self.WGy_plus, cond, x_init, a_init
            )

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
