from selector.meter_selector import register_metric
from utils.metrics import l1_metric, rmse_metric
from utils.util import AverageMeter, wrap_phase
import torch.nn.functional as F

@register_metric(name='MchMeter')
class MchMeter:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.epoch_meter = AverageMeter()
        self.mode = config.mode
        self.writer = config.writer
        self.epoch = 0
        self.acc_step = 0

    def setup_data(self, batch_dict):
        self.gt = batch_dict["gt"].to(self.device)
        self.pred = batch_dict["pred"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.pred_unwrapped = batch_dict["pred_unwrapped"].to(self.device)
        self.gt_wrapped = batch_dict["wrapped"].to(self.device)
        self.pred_wrapped = wrap_phase(self.pred_unwrapped)

    def compute_batch_metric(self):
        noise_mse = F.mse_loss(self.pred, self.gt)
        unwrapped_l1_loss = F.l1_loss(self.gt_unwrapped, self.pred_unwrapped)
        unwrapped_rmse_loss = rmse_metric(self.pred_unwrapped, self.gt_unwrapped)
        unwrapped_nrmse_loss = unwrapped_rmse_loss / (self.gt_unwrapped.max() - self.gt_unwrapped.min() + 1e-8)
        wrapped_l1_loss = F.l1_loss(self.gt_wrapped, self.pred_wrapped)
        wrapped_rmse_loss = rmse_metric(self.pred_wrapped, self.gt_wrapped)

        self.batch_metric_dict = {'NoiseMSE': noise_mse,
                                  'UnwrappedL1': unwrapped_l1_loss, 'UnwrappedRMSE': unwrapped_rmse_loss,'UnwrappedNRMSE': unwrapped_nrmse_loss,
                                  'WrappedL1': wrapped_l1_loss, 'WrappedRMSE': wrapped_rmse_loss
                                  }
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_epoch", self.epoch)

    def _record_metrics(self, metrics, prefix, step):
        if self.config.io.use_tensorboard:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)