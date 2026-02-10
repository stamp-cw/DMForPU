import torch
# from run.sde import ScoreFN
# from utils.dice_score import dice_loss
import torch.nn.functional as F

import torch

# from diffusion.elucidate_diffusion import SpectralFeatureExtractorPretrained
from selector.loss_type_selector import register_loss_type, _LOSSTYPE
from utils.util import wrap_phase

# @register_loss_type(name='PURE')
# class PureLossType:
#     def __init__(self, config):
#         self.config = config
#         self.name = config.loss_type.name
#
#     def __call__(self, diffusion):
#         # unet pred noise diff
#         noise_pred = diffusion.noise_pred
#         noise = diffusion.noise
#         diff_loss = F.mse_loss(noise_pred, noise)
#         total_loss = diff_loss
#         return total_loss

@register_loss_type(name='PURE2')
class Pure2LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        # print(self.meter.batch_metric_dict)
        diff_loss = self.meter.batch_metric_dict['MSE']
        total_loss = diff_loss
        return total_loss

@register_loss_type(name='Pure')
class PureLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        diff_loss = self.meter.batch_metric_dict['NoiseMSE']
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

@register_loss_type(name='PHY11')
class PHY11LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def unwrap_l1_loss(self, gt_unwrapped, pred_unwrapped):
        # pred_wrapped = wrap_phase(pred_unwrapped)
        loss = F.l1_loss(pred_unwrapped, gt_unwrapped)
        return loss

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
        # wrapped = diffusion.wrapped
        # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        phys_loss = self.unwrap_l1_loss(diffusion.gt_unwrapped, diffusion.pred_unwrapped)

        total_loss = diff_loss + 0.5 * phys_loss

        return total_loss


@register_loss_type(name='MchLoss')
class MchLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        noise_mse_loss = self.meter.batch_metric_dict['NoiseMSE']
        phys_loss = self.meter.batch_metric_dict['UnwrappedL1']
        total_loss = noise_mse_loss + 0.5 * phys_loss
        return total_loss


@register_loss_type(name='WavLoss')
class WavLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        noise_mse_loss = self.meter.batch_metric_dict['NoiseMSE']
        # phys_loss = self.meter.batch_metric_dict['UnwrappedL1']
        charbonnier_loss = self.meter.batch_metric_dict['CharbonnierLoss']
        wav_loss = self.meter.batch_metric_dict['WavLoss']
        lge_loss = self.meter.batch_metric_dict['LGELoss']
        mae_loss = self.meter.batch_metric_dict['UnwrappedMAE']
        # total_loss = noise_mse_loss + charbonnier_loss + wav_loss + lge_loss
        # total_loss = noise_mse_loss + 0.5 * charbonnier_loss + 0.1 * wav_loss + 0.001 * lge_loss
        bleta1 = self.config.loss_type.bleta1
        bleta2 = self.config.loss_type.bleta2
        bleta3 = self.config.loss_type.bleta3
        total_loss = noise_mse_loss + bleta1 * charbonnier_loss + bleta2 * wav_loss + bleta3 * lge_loss
        # total_loss = noise_mse_loss + 0.5 * mae_loss
        # total_loss = noise_mse_loss + 0.5 * charbonnier_loss
        # total_loss = noise_mse_loss + 10 * charbonnier_loss + 0.1 * wav_loss + 0.001 * lge_loss
        return total_loss

@register_loss_type(name='MchGradLoss')
class MchGradLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        noise_mse_loss = self.meter.batch_metric_dict['NoiseMSE']
        # phys_loss2 = self.meter.batch_metric_dict['UnwrappedL1']
        phys_loss = self.meter.batch_metric_dict['UnwrappedGradL1']
        # total_loss = noise_mse_loss + 0.5 * phys_loss + 0.001 * phys_loss2
        total_loss = noise_mse_loss + 0.5 * phys_loss
        return total_loss


@register_loss_type(name='MchPuLoss')
class MchPuLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, diffusion):
        # noise_mse_loss = self.meter.batch_metric_dict['NoiseMSE']
        # phys_loss = self.meter.batch_metric_dict['UnwrappedL1']
        # total_loss = noise_mse_loss + 0.5 * phys_loss
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        r_loss = self.residue_loss(diffusion.wrapped, diffusion.pred_unwrapped)
        c_loss = self.cross_loss(diffusion.gt_k_mat_disc, diffusion.pred_k_mat_disc)
        l_one_loss = self.l1_loss(diffusion.gt_unwrapped, diffusion.pred_unwrapped)

        # print(f"diff_loss: {diff_loss.item()}, r_loss: {r_loss.item()}, c_loss: {c_loss.item()}, l_one_loss: {l_one_loss.item()}")

        total_loss = diff_loss + 0.1 * r_loss + 0.1 * c_loss + 0.1 * l_one_loss
        return total_loss


@register_loss_type(name='PHY12')
class PHY12LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def kmat_cont_neg_norm_l1_loss(self, gt_kmat_cont_neg_norm, pred_k_mat_cont_neg_norm):
        # pred_wrapped = wrap_phase(pred_unwrapped)
        loss = F.l1_loss(pred_k_mat_cont_neg_norm, gt_kmat_cont_neg_norm)
        return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # diffusion solution pred unwrapped phase
        # wrapped = diffusion.wrapped
        # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        phys_loss = self.kmat_cont_neg_norm_l1_loss(diffusion.gt_k_mat_cont_neg_norm, diffusion.pred_k_mat_cont_neg_norm)

        total_loss = diff_loss + 0.5 * phys_loss

        return total_loss


@register_loss_type(name='PHY13')
class PHY13LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    def neg_norm_l1_loss(self, gt_kmat_cont_neg_norm, pred_k_mat_cont_neg_norm):
        # pred_wrapped = wrap_phase(pred_unwrapped)
        loss = F.l1_loss(pred_k_mat_cont_neg_norm, gt_kmat_cont_neg_norm)
        return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # diffusion solution pred unwrapped phase
        # wrapped = diffusion.wrapped
        # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        phys_loss = self.neg_norm_l1_loss(diffusion.gt_unwrapped_neg_norm, diffusion.pred_unwrapped_neg_norm)

        total_loss = diff_loss + 0.5 * phys_loss

        return total_loss

@register_loss_type(name='PHY14')
class PHY13LossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        # self.lam_phys = config.loss_type.lam_phys

    # def neg_norm_kl_loss(self, gt_unwrapped_neg_norm, pred_unwrapped_neg_norm):
    #     loss = torch.mean(gt_kmat_cont_neg_norm * torch.log((gt_kmat_cont_neg_norm + 1e-8) / (pred_k_mat_cont_neg_norm + 1e-8)))
    #     return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        noise_pred = diffusion.noise_pred
        noise = diffusion.noise
        diff_loss = F.mse_loss(noise_pred, noise)

        # diffusion solution pred unwrapped phase
        # wrapped = diffusion.wrapped
        # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        phys_loss = self.neg_norm_l1_loss(diffusion.gt_unwrapped_neg_norm, diffusion.pred_unwrapped_neg_norm)

        total_loss = diff_loss + 0.5 * phys_loss

        return total_loss

def SAM(x, y):
    cosine_sim = F.cosine_similarity(x, y, dim=1)
    #remove nan values
    if torch.isnan(cosine_sim).any():
        print('nan values in cosine_sim')
        cosine_sim[torch.isnan(cosine_sim)] = 1e-6
    if torch.isnan(x).any():
        print('nan values in x')
        x[torch.isnan(x)] = 1e-6
    return torch.acos(cosine_sim.clamp(-1+1e-6, 1-1e-6)).mean()

# @register_loss_type(name='GEWLOSS')
# class GEWLOSSType:
#     def __init__(self, config):
#         self.config = config
#         self.name = config.loss_type.name
#         # self.lam_phys = config.loss_type.lam_phys
#         self.perceptual_loss = SpectralFeatureExtractorPretrained(in_channels=1)
#
#     # def neg_norm_kl_loss(self, gt_unwrapped_neg_norm, pred_unwrapped_neg_norm):
#     #     loss = torch.mean(gt_kmat_cont_neg_norm * torch.log((gt_kmat_cont_neg_norm + 1e-8) / (pred_k_mat_cont_neg_norm + 1e-8)))
#     #     return loss
#
#     def gradient_loss(self,x_generated, x_real):
#         """ Compute gradient consistency loss """
#         c = x_generated.shape[1]
#         device = x_generated.device
#         dtype = x_generated.dtype
#
#         # Create a Sobel filter (1, 1, 3, 3)
#         sobel_x = torch.tensor(
#             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device
#         ).view(1, 1, 3, 3)
#
#         sobel_y = sobel_x.transpose(2, 3)  # y direction Sobel
#
#         # Copy to multichannel to make it work with `groups=c`
#         sobel_x = sobel_x.repeat(c, 1, 1, 1)
#         sobel_y = sobel_y.repeat(c, 1, 1, 1)
#
#         grad_x_gen = F.conv2d(x_generated, sobel_x, padding=1, groups=c)
#         grad_y_gen = F.conv2d(x_generated, sobel_y, padding=1, groups=c)
#         grad_x_real = F.conv2d(x_real, sobel_x, padding=1, groups=c)
#         grad_y_real = F.conv2d(x_real, sobel_y, padding=1, groups=c)
#         # Calculating L1 loss
#         loss = F.l1_loss(grad_x_gen, grad_x_real) + F.l1_loss(grad_y_gen, grad_y_real)
#         loss = loss / 2  # Divide by 2 to make it smoother
#         return loss

    def __call__(self, diffusion):
        # unet pred noise diff
        # noise_pred = diffusion.noise_pred
        # noise = diffusion.noise
        # diff_loss = F.mse_loss(noise_pred, noise)
        #
        # # diffusion solution pred unwrapped phase
        # # wrapped = diffusion.wrapped
        # # pred_wrapped = wrap_phase(diffusion.pred_unwrapped)
        # # phys_loss = F.l1_loss(pred_wrapped, wrapped)
        # phys_loss = self.neg_norm_l1_loss(diffusion.gt_unwrapped_neg_norm, diffusion.pred_unwrapped_neg_norm)

        # total_loss = diff_loss + 0.5 * phys_loss


        denoised = diffusion.pred_unwrapped_neg_norm
        images = diffusion.gt_unwrapped_norm


        pixel_loss = 0.5*F.mse_loss(denoised, images) + 0.5*SAM(denoised, images)
        # perception_loss = self.perceptual_loss(denoised, images)
        geometric_loss = self.gradient_loss(denoised, images)

        # total_loss = 0.8*pixel_loss + 0.1*perception_loss + 0.1*geometric_loss
        total_loss = 0.8*pixel_loss +  0.1*geometric_loss

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

@register_loss_type(name='VAEKLLOSS')
class VAEKLLossType:
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

@register_loss_type(name='VAEPURELOSS')
class VAEPURELossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name

    def __call__(self, vae):
        # unet pred noise diff
        recon_loss = F.l1_loss(vae.pred, vae.gt)
        kl_loss = vae.posterior.kl().mean()

        # kl_weight = min(1e-6, step / 10000 * 1e-6)
        # total_loss = recon_loss + kl_weight * kl_loss
        total_loss = recon_loss
        return total_loss


@register_loss_type(name='UNetLOSS')
class UNetLOSSLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name

    def __call__(self, vae):
        # unet pred noise diff
        recon_loss = F.l1_loss(vae.pred, vae.gt)
        total_loss = recon_loss
        return total_loss


@register_loss_type(name='PUNetLoss')
class PUNetLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, vae):
        # recon_loss = self.meter.batch_metric_dict['L1']
        recon_loss = self.meter.batch_metric_dict['MSE']
        total_loss = recon_loss
        return total_loss


@register_loss_type(name='DLPULoss')
class DLPULossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, vae):
        recon_loss = self.meter.batch_metric_dict['L1']
        total_loss = recon_loss
        return total_loss

@register_loss_type(name='SqdLstmLoss')
class SqdLstmLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, vae):
        var_loss = self.meter.batch_metric_dict['VAR']
        tv_loss = self.meter.batch_metric_dict['TV']
        total_loss =  var_loss + 0.1 * tv_loss
        return total_loss

@register_loss_type(name='RestormerLoss')
class RestormerLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, vae):
        recon_loss = self.meter.batch_metric_dict['L1']
        total_loss = recon_loss
        return total_loss


@register_loss_type(name='UformerLoss')
class UformerLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def __call__(self, vae):
        charbonnier_loss = self.meter.batch_metric_dict['CharbonnierLoss']
        total_loss = charbonnier_loss
        return total_loss

# @register_loss_type(name='U3NetLoss')
# class U3NetLossType:
#     def __init__(self, config):
#         self.config = config
#         self.name = config.loss_type.name
#         self.meter = config.train_meter
#
#     def __call__(self, vae):
#         charbonnier_loss = self.meter.batch_metric_dict['CharbonnierLoss']
#         total_loss = charbonnier_loss
#         return total_loss


@register_loss_type(name='U3NetLoss')
class U3NetLossType:
    def __init__(self, config):
        self.config = config
        self.name = config.loss_type.name
        self.meter = config.train_meter

    def Loss_SR(self,  target_grad, pred):
        pred_grad = self.gradient(pred)
        error = self.torch_Wrap(target_grad - pred_grad)
        loss = torch.mean(torch.square(error))
        return loss

    def gradient(self, x):
        res = torch.zeros(*x.shape, 2).type_as(x)
        res[:, :, :, 1:, 0] = x[..., 1:] - x[..., :-1]
        res[:, :, 1:, :, 1] = x[:, :, 1:, :] - x[:, :, :-1, :]
        return res

    def torch_Wrap(self, x):
        return torch.remainder(x+torch.pi, torch.ones_like(x) * (2*torch.pi))-torch.pi

    def __call__(self, mmodel):
        # charbonnier_loss = self.meter.batch_metric_dict['CharbonnierLoss']
        WGy_minus  = mmodel.WGy_minus
        L_sr = 0
        for j, each_x in enumerate(mmodel.pred_list):
            # L_sr += self.Loss_SR(WGy_minus, each_x) / (len(mmodel.pred_list) - j)
            # L_sr += self.Loss_SR(WGy_minus, each_x) / (3 - j) + 0.0 * each_x.mean()
            L_sr += self.Loss_SR(WGy_minus, each_x) / (3 - j)

        total_loss = L_sr
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