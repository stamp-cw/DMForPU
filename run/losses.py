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

@register_loss_type(name='PURE')
class PureLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)
        total_loss = diff_loss
        return total_loss

@register_loss_type(name='PHY1')
class PHY1LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def wrap_l1_loss(self, gt_wrapped, pred_unwrapped):
        pred_wrapped = wrap_phase(pred_unwrapped)
        loss = F.l1_loss(pred_wrapped, gt_wrapped)
        return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # diffusion solution pred unwrapped phase
        wrapped = diffusion.wrapped
        pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        phys_loss = F.l1_loss(pred_wrapped, wrapped)

        total_loss = diff_loss + 0.5 * phys_loss

        return total_loss

@register_loss_type(name='PHY2')
class PHY2LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

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

        l1_loss = F.l1_loss(diffusion.pred_wrapped, diffusion.wrapped)

        grad_loss = self.gradient_loss(diffusion.pred_wrapped, diffusion.wrapped)
        tv_loss = self.tv_loss(diffusion.pred_wrapped)

        # total_loss = diff_loss + self.lam_phys * phys_loss + 0.1 * grad_loss + 0.01 * tv_loss
        total_loss = diff_loss + 0.001 * l1_loss + 0.01 * grad_loss + 0.001 * tv_loss

        return total_loss

# PHNet2.0 loss
@register_loss_type(name='PHY3')
class PHY3LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def residue_loss(self, gt_wrapped, pred_unwrapped):
        gt_wrapped_grad_x = gt_wrapped[:, :, 1:, :] - gt_wrapped[:, :, :-1, :]  # x-axis gradient
        gt_wrapped_grad_y = gt_wrapped[:, :, :, 1:] - gt_wrapped[:, :, :, :-1]  # y-axis gradient
        pred_unwrapped_grad_x = pred_unwrapped[:, :, 1:, :] - pred_unwrapped[:, :, :-1, :]  # x-axis gradient
        pred_unwrapped_grad_y = pred_unwrapped[:, :, :, 1:] - pred_unwrapped[:, :, :, :-1]  # y-axis gradient
        # print(f"gt_wrapped_grad_x shape: {gt_wrapped_grad_x.shape}, pred_unwrapped_grad_x shape: {pred_unwrapped_grad_x.shape}")
        # print(f"gt_wrapped_grad_y shape: {gt_wrapped_grad_y.shape}, pred_unwrapped_grad_y shape: {pred_unwrapped_grad_y.shape}")
        loss = torch.mean(torch.abs(pred_unwrapped_grad_x - gt_wrapped_grad_x)) + torch.mean(torch.abs(pred_unwrapped_grad_y - gt_wrapped_grad_y))
        return loss

    def cross_loss(self, gt_k_mat, pred_k_mat):
        pred_k_mat = torch.clamp(pred_k_mat, min=1e-8)
        loss = -torch.mean(gt_k_mat * torch.log(pred_k_mat))
        return loss

    def l1_loss(self, gt_unwrapped, pred_unwrapped):
        loss = F.l1_loss(pred_unwrapped, gt_unwrapped)
        return loss


    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        r_loss = self.residue_loss(diffusion.wrapped, diffusion.pred_unwrapped)
        c_loss = self.cross_loss(diffusion.gt_k_mat_disc, diffusion.pred_k_mat_disc)
        l_one_loss = self.l1_loss(diffusion.gt_unwrapped, diffusion.pred_unwrapped)

        # print(f"diff_loss: {diff_loss.item()}, r_loss: {r_loss.item()}, c_loss: {c_loss.item()}, l_one_loss: {l_one_loss.item()}")

        total_loss = diff_loss + 0.1 * r_loss + 0.1 * c_loss + 0.1 * l_one_loss

        return total_loss

# PHNet2.0 loss
@register_loss_type(name='PHY4')
class PHY4LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def residue_loss(self, gt_wrapped, pred_unwrapped):
        gt_wrapped_grad_x = gt_wrapped[:, :, 1:, :] - gt_wrapped[:, :, :-1, :]  # x-axis gradient
        gt_wrapped_grad_y = gt_wrapped[:, :, :, 1:] - gt_wrapped[:, :, :, :-1]  # y-axis gradient
        pred_unwrapped_grad_x = pred_unwrapped[:, :, 1:, :] - pred_unwrapped[:, :, :-1, :]  # x-axis gradient
        pred_unwrapped_grad_y = pred_unwrapped[:, :, :, 1:] - pred_unwrapped[:, :, :, :-1]  # y-axis gradient
        # print(f"gt_wrapped_grad_x shape: {gt_wrapped_grad_x.shape}, pred_unwrapped_grad_x shape: {pred_unwrapped_grad_x.shape}")
        # print(f"gt_wrapped_grad_y shape: {gt_wrapped_grad_y.shape}, pred_unwrapped_grad_y shape: {pred_unwrapped_grad_y.shape}")
        loss = torch.mean(torch.abs(pred_unwrapped_grad_x - gt_wrapped_grad_x)) + torch.mean(torch.abs(pred_unwrapped_grad_y - gt_wrapped_grad_y))
        return loss

    def cross_loss(self, gt_k_mat, pred_k_mat):
        pred_k_mat = torch.clamp(pred_k_mat, min=1e-8)
        loss = -torch.mean(gt_k_mat * torch.log(pred_k_mat))
        return loss

    def wrap_l1_loss(self, gt_wrapped, pred_unwrapped):
        pred_wrapped = wrap_phase(pred_unwrapped)
        loss = F.l1_loss(pred_wrapped, gt_wrapped)
        return loss

    def l1_loss(self, gt_unwrapped, pred_unwrapped):
        loss = F.l1_loss(pred_unwrapped, gt_unwrapped)
        return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        r_loss = self.residue_loss(diffusion.wrapped, diffusion.pred_unwrapped)
        # c_loss = self.cross_loss(diffusion.gt_k_mat_disc, diffusion.pred_k_mat_disc)
        l_one_loss = self.l1_loss(diffusion.gt_unwrapped, diffusion.pred_unwrapped)
        wrap_l_one_loss = self.wrap_l1_loss(diffusion.wrapped, diffusion.pred_unwrapped)

        # print(f"diff_loss: {diff_loss.item()}, r_loss: {r_loss.item()}, c_loss: {c_loss.item()}, l_one_loss: {l_one_loss.item()}, wrap_l_one_loss: {wrap_l_one_loss.item()}")

        # total_loss = diff_loss + 0.05 * r_loss + 0.1 * c_loss + 0.1 * l_one_loss
        total_loss = diff_loss + 0.1 * r_loss + 0.1 * wrap_l_one_loss + 0.1 * l_one_loss

        return total_loss

@register_loss_type(name='VAELOSS')
class VAELossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name

    def __call__(self, vae):
        # unet pred noise diff
        recon_loss = F.l1_loss(vae.pred, vae.gt)
        kl_loss = vae.posterior.kl().mean()

        # kl_weight = min(1e-6, step / 10000 * 1e-6)
        # total_loss = recon_loss + kl_weight * kl_loss
        total_loss = recon_loss + 0.0001 * kl_loss
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
