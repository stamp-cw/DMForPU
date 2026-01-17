import torch
# from run.sde import ScoreFN
# from utils.dice_score import dice_loss
import torch.nn.functional as F

import torch
from selector.loss_type_selector import register_loss_type, _LOSSTYPE
from utils.util import wrap_phase


# def compute_losses(noise_pred: torch.Tensor, noise: torch.Tensor, scheduler: "DDPMScheduler", t: torch.Tensor, noisy: torch.Tensor, wrapped: torch.Tensor, lambda_phys: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Compute diffusion MSE and physics-consistency L1 loss."""
#     diff_loss = F.mse_loss(noise_pred, noise)
#     pred_unwrapped = scheduler.step(noise_pred, t, noisy).prev_sample
#     phys_loss = F.l1_loss(wrap_phase(pred_unwrapped), wrapped)
#     total = diff_loss + lambda_phys * phys_loss
#     return total, {"diff": diff_loss.detach(), "phys": phys_loss.detach()}

@register_loss_type(name='PHY')
class PHYLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.lam_phys = config.loss_type.lam_phys


    def gradient_loss(self, pred, target):
        grad_pred_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])  # x-axis gradient
        grad_pred_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])  # y-axis gradient
        grad_target_x = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])  # x-axis gradient
        grad_target_y = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])  # y-axis gradient

        grad_loss_x = torch.mean(grad_pred_x - grad_target_x)
        grad_loss_y = torch.mean(grad_pred_y - grad_target_y)

        return grad_loss_x + grad_loss_y


    def tv_loss(self, x):
        # 各向异性 Total Variation (TV)
        tv_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # x-direction gradient
        tv_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # y-direction gradient
        return torch.sum(tv_x) + torch.sum(tv_y)

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # diffusion solution pred unwrapped phase
        # wrapped = diffusion.wrapped
        # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        #

        phys_loss = F.mse_loss(diffusion.pred_k_mat_cont_neg_norm, diffusion.gt_k_mat_cont_neg_norm)

        # grad_loss = self.gradient_loss(pred_wrapped, wrapped)
        # tv_loss = self.tv_loss(pred_wrapped)

        # total_loss = diff_loss + self.lam_phys * phys_loss + 0.1 * grad_loss + 0.01 * tv_loss
        # total_loss = diff_loss + 0.001 * phys_loss + 0.01 * grad_loss + 0.001 * tv_loss
        total_loss = diff_loss + phys_loss

        return total_loss

class LossFN:
    def __init__(self, config):
        self.config = config
        self.loss_type_fn = _LOSSTYPE(self.config)

    def __call__(self, diffusion):
        return self.loss_fn(diffusion)

    # def loss_fn(self, model, batch):
    def loss_fn(self, diffusion):
        losses = self.loss_type_fn(diffusion)
        loss = losses
        return loss
