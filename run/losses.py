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

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # # diffusion solution pred unwrapped phase
        # wrapped_norm = diffusion.wrapped / torch.pi
        # pred_wrapped_norm = wrap_phase(diffusion.pred_unwrapped) / torch.pi
        # phys_loss = F.l1_loss(pred_wrapped_norm, wrapped_norm)

        # total_loss = diff_loss + self.lam_phys * phys_loss
        total_loss = diff_loss

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
