import torch

from selector.meter_selector import register_metric
from utils.util import AverageMeter, wrap_phase
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

    def setup_data(self, batch_dict):
        self.gt = batch_dict["gt"].to(self.device)
        self.pred = batch_dict["pred"].to(self.device)

    def compute_batch_metric(self):
        l1_loss = F.l1_loss(self.gt, self.pred)
        self.batch_metric_dict = {'L1': l1_loss}
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_epoch", self.epoch)

    def _record_metrics(self, metrics, prefix, step):
        if self.config.io.use_tensorboard:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)