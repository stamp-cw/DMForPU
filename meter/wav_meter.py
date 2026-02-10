import torch
import wandb
from tornado.locale import load_gettext_translations

from selector.meter_selector import register_metric
from utils.metrics import l1_metric, rmse_metric
from utils.util import AverageMeter, wrap_phase, multi_scale_wavelet, MultiScaleWavelet
import torch.nn.functional as F
import wandb

@register_metric(name='WavMeter')
class WavMeter:
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
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.pred_unwrapped = batch_dict["pred_unwrapped"].to(self.device)
        self.gt_wrapped = batch_dict["wrapped"].to(self.device)
        self.pred_wrapped = wrap_phase(self.pred_unwrapped)

    def charbonnier_loss(self, pred, gt):
        eps = 1e-3
        diff = pred - gt
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (eps*eps)))
        return loss

    def wav_loss(self, pred, gt):
        '''多尺度小波分解损失'''
        wavelet_layer = MultiScaleWavelet(wavelet='db4', level=2)
        gt_wav = wavelet_layer.decompose(pred.detach())
        pred_wav = wavelet_layer.decompose(gt.detach())
        wav_loss = F.l1_loss(gt_wav, pred_wav)
        return wav_loss

    def local_gradient_energy_loss(self, pred, gt):
        num_regions = 8
        B, C, H, W = pred.shape
        region_width = W // num_regions
        lge_loss = 0.0
        for i in range(num_regions):
            start = i * region_width
            end = (i+1)*region_width if i < num_regions-1 else W
            pred_block = pred[:, :, :, start:end]
            gt_block = gt[:, :, :, start:end]
            gx_pred, gy_pred = self.phase_gradient_torch(pred_block)
            gx_gt, gy_gt = self.phase_gradient_torch(gt_block)
            energy_pred = gx_pred**2 + gy_pred**2
            energy_gt   = gx_gt**2 + gy_gt**2
            # lge_loss += F.mse_loss(energy_gt, energy_pred)
            lge_loss += F.l1_loss(energy_gt, energy_pred)
        lge_loss = lge_loss / num_regions
        return lge_loss

    def pge_loss(self, pred, gt):
        gx_pred, gy_pred = self.phase_gradient_torch(pred)
        gx_gt, gy_gt = self.phase_gradient_torch(gt)
        pge_loss = (F.l1_loss(gx_pred, gx_gt) + F.l1_loss(gy_pred, gy_gt)) / 2
        return pge_loss

    def compute_batch_metric(self):
        # Loss Metric
        noise_mse = F.mse_loss(self.pred, self.gt)
        charbonnier_loss = self.charbonnier_loss(self.pred_unwrapped, self.gt_unwrapped)
        wav_loss = self.wav_loss(self.pred_unwrapped, self.gt_unwrapped)
        lge_loss = self.local_gradient_energy_loss(self.pred_unwrapped, self.gt_unwrapped)

        # Eval Metric
        wrapped_l1_loss = F.l1_loss(self.gt_wrapped, self.pred_wrapped)
        wrapped_rmse_loss = rmse_metric(self.pred_wrapped, self.gt_wrapped)

        unwrapped_l1_loss = F.l1_loss(self.gt_unwrapped, self.pred_unwrapped)
        unwrapped_rmse_loss = rmse_metric(self.pred_unwrapped, self.gt_unwrapped)
        unwrapped_nrmse_loss = unwrapped_rmse_loss / (self.gt_unwrapped.max() - self.gt_unwrapped.min() + 1e-8)
        unwrapped_pge_loss =  self.pge_loss(self.pred_unwrapped, self.gt_unwrapped)

        self.batch_metric_dict = {
            # Loss Metric
            'NoiseMSE': noise_mse,
            'CharbonnierLoss': charbonnier_loss,
            'WavLoss': wav_loss,
            'LGELoss': lge_loss,
            # Eval Metric
            'UnwrappedL1': unwrapped_l1_loss,
            'UnwrappedMAE': unwrapped_l1_loss, 'UnwrappedPGE': unwrapped_pge_loss,
            'UnwrappedRMSE': unwrapped_rmse_loss,'UnwrappedNRMSE': unwrapped_nrmse_loss,
            'WrappedL1': wrapped_l1_loss, 'WrappedRMSE': wrapped_rmse_loss,
        }
        self._record_metrics(self.batch_metric_dict, f"{self.mode}_per_batch", self.acc_step)

    def compute_epoch_metric(self):
        self.epoch_metric_dict = self.epoch_meter.avg()
        self.epoch_meter.reset()
        self._record_metrics(self.epoch_metric_dict, f"{self.mode}_per_epoch", self.epoch, is_epoch=True)

    def _record_metrics(self, metrics, prefix, step , is_epoch=False):
        if self.is_record:
            if self.config.io.use_tensorboard:
                for k, v in metrics.items():
                    self.writer.add_scalar(f"{prefix}/{k}", v, step)
            if self.config.io.use_wandb and is_epoch:
                # for k, v in metrics.items():
                #     wandb.log({f"{prefix}/{k}": v}, step=step+1, commit=True)
                # wandb.log({f"{prefix}": metrics}, step=step+1, commit=True)
                new_metric = {f"{prefix}/{k}": v for k, v in metrics.items()}
                wandb.log(new_metric, step=step+1, commit=True)

    def phase_gradient_torch(self, phase):
        """
        Args:
            phase: torch.Tensor, shape (H, W) or (B, 1, H, W)

        Returns:
            gx, gy: real-valued gradients
        """

        # gx = phase[..., :, 1:] - phase[..., :, :-1]
        # gy = phase[..., 1:, :] - phase[..., :-1, :]
        phase.unsqueeze(0)

        kx = torch.tensor(
            [[0, 0, 0],
             [-1, 1, 0],
             [0, 0, 0]],
            device=phase.device, dtype=phase.dtype
        ).view(1, 1, 3, 3)

        gx = F.conv2d(phase, kx, padding=1, groups=1, stride=1)
        gx[..., :, 0] = gx[..., :, 1] - (gx[..., :, 2] - gx[..., :, 1])

        ky = torch.tensor(
            [[0, -1, 0],
             [0,  1, 0],
             [0,  0, 0]],
            device=phase.device, dtype=phase.dtype
        ).view(1, 1, 3, 3)
        gy = F.conv2d(phase, ky, padding=1, groups=1, stride=1)
        gy[..., 0, :] = gy[..., 1, :] - (gy[..., 2, :] - gy[..., 1, :])

        return gx, gy