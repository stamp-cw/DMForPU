import torch
import wandb

from selector.meter_selector import register_metric
from utils.metrics import rmse_metric
from utils.util import AverageMeter, wrap_phase, phase_gradient_torch
import torch.nn.functional as F

@register_metric(name='RestormerMeter')
class RestormerMeter:
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

    def compute_batch_metric(self):
        l1_loss = F.l1_loss(self.gt, self.pred)
        mae_loss = F.l1_loss(self.gt, self.pred)
        rmse_loss = rmse_metric(self.pred, self.gt)
        nrmse_loss = rmse_loss / (self.gt.max() -self.gt.min() + 1e-8)
        # self.batch_metric_dict = {'L1': l1_loss,
        #                             'MAE': mae_loss,
        #                                 'RMSE': rmse_loss,'NRMSE':nrmse_loss
        #                           }
        mse_loss = F.mse_loss(self.gt, self.pred)
        gt_gx, gt_gy = phase_gradient_torch(self.gt)
        pred_gx, pred_gy = phase_gradient_torch(self.pred)
        pge_loss = F.l1_loss(torch.concat([gt_gx, gt_gy],dim=1), torch.concat([pred_gx, pred_gy],dim=1))
        self.batch_metric_dict = {
            'L1': l1_loss,
            'MSE': mse_loss,
            'MAE': mae_loss,
            'RMSE': rmse_loss,
            'NRMSE':nrmse_loss,
            'PGE': pge_loss
        }
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_epoch", self.epoch)

    def _record_metrics(self, metrics, prefix, step , is_epoch=False):
        if self.is_record:
            if self.config.io.use_tensorboard:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"{prefix}/{k}", v, step )
            if self.config.io.use_wandb and is_epoch:
                new_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
                wandb.log(new_metrics, commit=True)