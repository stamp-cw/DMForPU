from selector.meter_selector import register_metric
from utils.metrics import l1_metric, rmse_metric
from utils.util import AverageMeter
import torch.nn.functional as F

@register_metric(name='PUREMeter')
class PureMeter:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device
        self.epoch_meter = AverageMeter()
        self.mode = config.mode
        self.writer = config.writer
        self.epoch = 0

    def setup_data(self, batch_dict):
        self.gt = batch_dict["gt"].to(self.device)
        self.pred = batch_dict["pred"].to(self.device)

    def compute_batch_metric(self):
        mse = F.mse_loss(self.pred, self.gt)
        l_one = l1_metric(self.pred, self.gt)
        rmse = rmse_metric(self.pred, self.gt)
        self.batch_metric_dict = {'MSE': mse, 'L1': l_one, 'RMSE': rmse}
        self.epoch_meter.update(self.batch_metric_dict)
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.config.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_batch", self.epoch)

    def _record_metrics(self, metrics, prefix, step):
        if self.config.io.use_tensorboard:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)