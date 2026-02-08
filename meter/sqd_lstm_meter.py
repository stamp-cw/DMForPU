import torch

from selector.meter_selector import register_metric
from utils.metrics import rmse_metric
from utils.util import AverageMeter, wrap_phase
import torch.nn.functional as F

@register_metric(name='SqdLstmMeter')
class SqdLstmMeter:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.epoch_meter = AverageMeter()
        self.mode = config.mode
        self.writer = config.writer
        self.epoch = 0
        self.acc_step = 0
        self.is_record = True

    def setup_data(self, batch_dict):
        self.gt = batch_dict["gt"].to(self.device)
        self.pred = batch_dict["pred"].to(self.device)


    def tv_loss(self, y_true, y_pred):
        """
        Total Variation Loss (TV loss) between y_true and y_pred.
        Works for arbitrary height and width.

        Args:
            y_true: [B, C, H, W]
            y_pred: [B, C, H, W]
        Returns:
            L_tv: scalar loss
        """
        # 计算 x 方向差分
        y_x = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
        y_bar_x = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

        # 计算 y 方向差分
        y_y = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
        y_bar_y = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]

        # TV loss
        L_tv = torch.mean(torch.abs(y_x - y_bar_x)) + torch.mean(torch.abs(y_y - y_bar_y))
        return L_tv

    def var_loss(self, y_true, y_pred):
        """
        Variance Loss (通用版)
        Computes:
            L_var = mean( E^2 ) - mean(E)^2
        where E = y_pred - y_true

        Args:
            y_true: [B, C, H, W]
            y_pred: [B, C, H, W]

        Returns:
            L_var: scalar
        """
        E = y_pred - y_true
        # 先对每个样本计算 batch 内各元素的均值
        mean_square = torch.mean(E ** 2, dim=(1, 2, 3))  # [B]
        square_mean = torch.mean(E, dim=(1, 2, 3)) ** 2   # [B]
        # 取 batch 平均
        L_var = torch.mean(mean_square - square_mean)
        return L_var

    def compute_batch_metric(self):
        l_tv_loss = self.tv_loss(y_true=self.gt, y_pred=self.pred)
        l_var_loss = self.var_loss(y_true=self.gt, y_pred=self.pred)
        mae_loss = F.l1_loss(self.gt, self.pred)
        rmse_loss = rmse_metric(self.pred, self.gt)
        nrmse_loss = rmse_loss / (self.gt.max() - self.gt.min() + 1e-8)
        self.batch_metric_dict = {'L1': mae_loss,
                                  'TV': l_tv_loss, 'VAR': l_var_loss,
                                  'MAE': mae_loss, 'RMSE': rmse_loss, 'NRMSE': nrmse_loss}
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_epoch", self.epoch)

    def _record_metrics(self, metrics, prefix, step):
        if self.is_record:
            if self.config.io.use_tensorboard:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"{prefix}/{k}", v, step)