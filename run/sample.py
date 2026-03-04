from functools import cached_property

import torch

from diffusion.diffusion_setup import DiffusionSetup
from selector.data_selector import _DATA_LOADERS
import matplotlib.pyplot as plt
from matplotlib import colors

class Sampler:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = config.sampling.device
        self.diffusion = DiffusionSetup(self.config, self.logger).diffusion
        self.data_loader = _DATA_LOADERS(self.config)

    def sample(self):
        sampling_iter = iter(self.sampling_loader)
        if self.config.sampling.fix_seed:
            torch.manual_seed(self.config.seed)  # 设置 CPU 随机数种子
            torch.cuda.manual_seed_all(self.config.seed)  # 设置所有 GPU 随机数种子

        self.logger.info(f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        self.diffusion.setup_eval()
        while self.remaining_samples > 0:
            try:
                batch_dict = next(sampling_iter)
            except StopIteration:
                sampling_iter = iter(self.sampling_loader)
                batch_dict = next(sampling_iter)
            self._sample(batch_dict)
            self._save_samples_png()
            self._update_stat()

    def _sample(self, batch_dict):
        with torch.no_grad():
            self.diffusion.setup_data(batch_dict)
            self.diffusion.infer_sample()
            # self._save_samples_and_preview(wrapped, gt_unwrapped, pred_unwrapped)
            c_batch = {}
            if self.config.diffusion.name == 'KdfDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.diff_unwrapped,
                    "gt_k_mat_cont_neg_norm": batch_dict["k_mat_cont_neg_norm"].to(self.device),
                    "pred_k_mat_cont_neg_norm": self.diffusion.pred_k_mat_cont_neg_norm,
                    "diff_k_mat_cont_neg_norm": batch_dict["k_mat_cont_neg_norm"].to(self.device) - self.diffusion.pred_k_mat_cont_neg_norm ,
                    "gt_k_mat_cont": batch_dict["k_mat_cont"].to(self.device),
                    "pred_k_mat_cont": self.diffusion.pred_k_mat_cont,
                    "diff_k_mat_cont": batch_dict["k_mat_cont"].to(self.device) - self.diffusion.pred_k_mat_cont ,
                    "gt_k_mat_disc": batch_dict["k_mat_disc"].to(self.device),
                    "pred_k_mat_disc": self.diffusion.pred_k_mat_disc,
                    "diff_k_mat_disc": batch_dict["k_mat_disc"].to(self.device) - self.diffusion.pred_k_mat_disc ,
                }
                self._save_compare_png(c_batch)
            # elif self.config.diffusion.name == 'DFNDDPMDiffusion':
            #     c_batch = {
            #         "wrapped": self.diffusion.wrapped,
            #         "gt_unwrapped": self.diffusion.gt_unwrapped,
            #         "pred_unwrapped": self.diffusion.pred_unwrapped,
            #         "diff_unwrapped": self.diffusion.diff_unwrapped,
            #         "gt_unwrapped_sub_wrapped_neg_norm": batch_dict["unwrapped_sub_wrapped_neg_norm"].to(self.device),
            #         "pred_unwrapped_sub_wrapped_neg_norm": self.diffusion.pred_unwrapped_sub_wrapped_neg_norm,
            #         "diff_unwrapped_sub_wrapped_neg_norm": batch_dict["unwrapped_sub_wrapped_neg_norm"].to(self.device) - self.diffusion.pred_unwrapped_sub_wrapped_neg_norm ,
            #     }
            #     self._save_compare_png_dfn(c_batch)
            elif self.config.diffusion.name == 'NegNormDDPMDiffusion' or self.config.diffusion.name == 'DDPMDiffusion' or self.config.diffusion.name == 'MchNegNormDDPMDiffusion' \
                    or self.config.diffusion.name == 'DphDDPMDiffusion' or self.config.diffusion.name == 'ElucidatedDiffusion' or self.config.diffusion.name == 'MchDDPMDiffusion'\
                    or self.config.diffusion.name == 'WavDDPMDiffusion' or self.config.diffusion.name == 'CfgDDPMDiffusion' or self.config.diffusion.name == 'DfnDDPMDiffusion' \
                    or self.config.diffusion.name == 'FduDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "wrapped_neg_norm": batch_dict["wrapped_neg_norm"].to(self.device),
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.gt_unwrapped - self.diffusion.pred_unwrapped,
                    "gt_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device),
                    "pred_unwrapped_neg_norm": self.diffusion.pred_unwrapped_neg_norm,
                    "diff_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device) - self.diffusion.pred_unwrapped_neg_norm
                }
                self._save_compare_png_neg_norm(c_batch)
            elif self.config.diffusion.name == 'LatentDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.diff_unwrapped,
                    "gt_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device),
                    "pred_unwrapped_neg_norm": self.diffusion.pred_unwrapped_neg_norm,
                    "diff_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device) - self.diffusion.pred_unwrapped_neg_norm
                }
                self._save_compare_png_latent(c_batch)
            elif self.config.diffusion.name == 'UcnDDPMDiffusion':
                c_batch = {
                    "gt_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device),
                    "pred_unwrapped_neg_norm": self.diffusion.pred_unwrapped_neg_norm,
                }
                self._save_compare_png_ucn(c_batch)
            elif self.config.diffusion.name == 'PhaseDDPMDiffusion' or self.config.diffusion.name == 'PhaseCutDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "wrapped_neg_norm": batch_dict["wrapped_neg_norm"].to(self.device),
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.gt_unwrapped - self.diffusion.pred_unwrapped,
                    "gt_unwrapped_std_norm": batch_dict["unwrapped_std_norm"].to(self.device),
                    "pred_unwrapped_std_norm": self.diffusion.pred_unwrapped_std_norm,
                    "diff_unwrapped_std_norm": batch_dict["unwrapped_std_norm"].to(self.device) - self.diffusion.pred_unwrapped_std_norm
                }
                self._save_compare_png_std_norm(c_batch)
            elif self.config.diffusion.name == 'GradDDPMDiffusion':
                gt_unwrapped_grad_x_neg_norm, gt_unwrapped_grad_y_neg_norm=torch.chunk(batch_dict["unwrapped_grad_neg_norm"].to(self.device), chunks=2, dim=1)
                pred_unwrapped_grad_x_neg_norm, pred_unwrapped_grad_y_neg_norm=torch.chunk(self.diffusion.pred_unwrapped_grad_neg_norm, chunks=2, dim=1)
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.gt_unwrapped - self.diffusion.pred_unwrapped,
                    "gt_unwrapped_grad_x_neg_norm": gt_unwrapped_grad_x_neg_norm,
                    "pred_unwrapped_grad_x_neg_norm": pred_unwrapped_grad_x_neg_norm,
                    "diff_unwrapped_grad_x_neg_norm": gt_unwrapped_grad_x_neg_norm - pred_unwrapped_grad_x_neg_norm,
                    "gt_unwrapped_grad_y_neg_norm": gt_unwrapped_grad_y_neg_norm,
                    "pred_unwrapped_grad_y_neg_norm": pred_unwrapped_grad_y_neg_norm,
                    "diff_unwrapped_grad_y_neg_norm": gt_unwrapped_grad_y_neg_norm - pred_unwrapped_grad_y_neg_norm,
                    "wrapped_neg_norm": batch_dict["wrapped_neg_norm"].to(self.device),
                    "gt_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device),
                    "pred_unwrapped_neg_norm": self.diffusion.pred_unwrapped_neg_norm,
                    "diff_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device) - self.diffusion.pred_unwrapped_neg_norm
                }
                self._save_compare_png_grad_neg_norm(c_batch)
            else:
                # c_batch = {}
                assert "NotImplementedError"

            self._save_compare_pt(c_batch)
            self.samples = self.diffusion.pred_unwrapped


    def _save_compare_png_grad_neg_norm(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped_neg_norm, gt_unwrapped_neg_norm, pred_unwrapped_neg_norm, diff_unwrapped_neg_norm = c_batch['wrapped_neg_norm'], c_batch['gt_unwrapped_neg_norm'], c_batch['pred_unwrapped_neg_norm'], c_batch['diff_unwrapped_neg_norm']
        wrapped_neg_norm, gt_unwrapped_grad_x_neg_norm, pred_unwrapped_grad_x_neg_norm, diff_unwrapped_grad_x_neg_norm = c_batch['wrapped_neg_norm'], c_batch['gt_unwrapped_grad_x_neg_norm'], c_batch['pred_unwrapped_grad_x_neg_norm'], c_batch['diff_unwrapped_grad_x_neg_norm']
        wrapped_neg_norm, gt_unwrapped_grad_y_neg_norm, pred_unwrapped_grad_y_neg_norm, diff_unwrapped_grad_y_neg_norm = c_batch['wrapped_neg_norm'], c_batch['gt_unwrapped_grad_y_neg_norm'], c_batch['pred_unwrapped_grad_y_neg_norm'], c_batch['diff_unwrapped_grad_y_neg_norm']
        # wrapped, gt_k_mat_cont, pred_k_mat_cont, diff_k_mat_cont = c_batch['wrapped'], c_batch['gt_k_mat_cont'], c_batch['pred_k_mat_cont'], c_batch['diff_k_mat_cont']
        # wrapped, gt_k_mat_disc, pred_k_mat_disc, diff_k_mat_disc = c_batch['wrapped'], c_batch['gt_k_mat_disc'], c_batch['pred_k_mat_disc'], c_batch['diff_k_mat_disc']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped neg_norm", "GT unwrapped_neg_norm", "Pred unwrapped_neg_norm", "Diff unwrapped_neg_norm",
                  "Wrapped neg_norm", "GT unwrapped_grad_x_neg_norm", "Pred unwrapped_grad_x_neg_norm", "Diff unwrapped_grad_x_neg_norm",
                  "Wrapped neg_norm", "GT unwrapped_grad_y_neg_norm", "Pred unwrapped_grad_y_neg_norm", "Diff unwrapped_grad_y_neg_norm",
                  # "Wrapped", "GT k-mat Cont", "Pred k-mat Cont", "Diff k-mat Cont",
                  # "Wrapped", "GT k-mat Disc", "Pred k-mat Disc", "Diff k-mat Disc"
                  ]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped_neg_norm[i]), _to_numpy_2d(gt_unwrapped_neg_norm[i]), _to_numpy_2d(pred_unwrapped_neg_norm[i]), _to_numpy_2d(diff_unwrapped_neg_norm[i]),
                _to_numpy_2d(wrapped_neg_norm[i]), _to_numpy_2d(gt_unwrapped_grad_x_neg_norm[i]), _to_numpy_2d(pred_unwrapped_grad_x_neg_norm[i]), _to_numpy_2d(diff_unwrapped_grad_x_neg_norm[i]),
                _to_numpy_2d(wrapped_neg_norm[i]), _to_numpy_2d(gt_unwrapped_grad_y_neg_norm[i]), _to_numpy_2d(pred_unwrapped_grad_y_neg_norm[i]), _to_numpy_2d(diff_unwrapped_grad_y_neg_norm[i])
            ]
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "inferno",
                     "twilight", "viridis", "viridis", "inferno",
                     "twilight", "viridis", "viridis", "inferno"
                     ]
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[:4]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[4:8]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[8:]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_pt(self, c_batch):
        pt_path = self.config.io.generated_compare_pt_file_path(self.saved_samples,
                                                               self.saved_samples + self.temp_batch_size)
        torch.save(c_batch, pt_path)
        self.logger.info(f"Saved {self.temp_batch_size} samples to {pt_path}")

    def _save_samples_png(self):
        samples = self.samples  # shape: [B, H, W] 或 [B, 1, H, W]
        for i, img_array in enumerate(samples):
            img = img_array.squeeze()  # [H, W]
            img = img.detach().cpu().numpy()
            img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1)
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap="turbo")
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close()
        self.logger.info(
            f"Saved {self.samples.shape[0]} samples as PNG images in folder: {self.config.io.out_raw_sample_path}")

    # def load_checkpoint(self):
    #     self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
    #     loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
    #     self.diffusion.model.load_state_dict(loaded_state['model'])
    #     # if self.config.diffusion.use_controlnet:
    #     #     self.diffusion.controlnet_model.load_state_dict(loaded_state['controlnet_model'])

    def load_checkpoint(self):
        ckpt_path = self.config.io.sampling_ckpt_file_path
        self.logger.info(f"Loading checkpoint from {ckpt_path}")

        # 加载 state_dict
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model' not in checkpoint:
            raise KeyError(f"'model' key not found in checkpoint {ckpt_path}")

        state_dict = checkpoint['model']
        state_dict2 = checkpoint['controlnet_model']

        # 判断是否是多卡权重
        is_multi_card = any(k.startswith("module.") for k in state_dict.keys())
        if is_multi_card:
            self.logger.info("Detected multi-card checkpoint. Stripping 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            state_dict2 = {k.replace("module.", "", 1): v for k, v in state_dict2.items()}
        else:
            self.logger.info("Detected single-card checkpoint.")

        # 加载到模型
        self.diffusion.model.load_state_dict(state_dict, strict=True)
        self.diffusion.controlnet_model.load_state_dict(state_dict2, strict=True)
        self.logger.info("Checkpoint loaded successfully.")

    # def load_checkpoint(self):
    #     ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
    #     model_ckpt = ckpt['model']
    #     optimizer_ckpt = ckpt['optimizer']
    #
    #     r_key = 'module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.module.'
    #     is_multi_card = any(k.startswith("module.") for k in model_ckpt.keys())
    #     if is_multi_card:
    #         self.logger.info("Detected multi-card checkpoint. Stripping 'module.' prefix...")
    #         model_ckpt = {k.replace(r_key, "", 1): v for k, v in model_ckpt.items()}
    #         optimizer_ckpt = {k.replace(r_key, "", 1): v for k, v in optimizer_ckpt.items()}
    #     else:
    #         self.logger.info("Detected single-card checkpoint.")
    #
    #
    #     # self.diffusion.model.load_state_dict(ckpt['model'])
    #     self.diffusion.model.load_state_dict(model_ckpt)
    #     # self.optimizer.load_state_dict(ckpt['optimizer'])
    #     # self.optimizer.load_state_dict(optimizer_ckpt)


    def _save_compare_png(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped, gt_k_mat_cont_neg_norm, pred_k_mat_cont_neg_norm, diff_k_mat_cont_neg_norm = c_batch['wrapped'], c_batch['gt_k_mat_cont_neg_norm'], c_batch['pred_k_mat_cont_neg_norm'], c_batch['diff_k_mat_cont_neg_norm']
        wrapped, gt_k_mat_cont, pred_k_mat_cont, diff_k_mat_cont = c_batch['wrapped'], c_batch['gt_k_mat_cont'], c_batch['pred_k_mat_cont'], c_batch['diff_k_mat_cont']
        wrapped, gt_k_mat_disc, pred_k_mat_disc, diff_k_mat_disc = c_batch['wrapped'], c_batch['gt_k_mat_disc'], c_batch['pred_k_mat_disc'], c_batch['diff_k_mat_disc']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped", "GT k-mat Cont NegNorm", "Pred k-mat Cont NegNorm", "Diff k-mat Cont NegNorm",
                  "Wrapped", "GT k-mat Cont", "Pred k-mat Cont", "Diff k-mat Cont",
                  "Wrapped", "GT k-mat Disc", "Pred k-mat Disc", "Diff k-mat Disc"]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_cont_neg_norm[i]), _to_numpy_2d(pred_k_mat_cont_neg_norm[i]), _to_numpy_2d(diff_k_mat_cont_neg_norm[i]),
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_cont[i]), _to_numpy_2d(pred_k_mat_cont[i]), _to_numpy_2d(diff_k_mat_cont[i]),
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_disc[i]), _to_numpy_2d(pred_k_mat_disc[i]), _to_numpy_2d(diff_k_mat_disc[i])
            ]
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "inferno",
                     "twilight", "viridis", "viridis", "inferno",
                     "twilight", "viridis", "viridis", "inferno"]
            zip_list = list(zip(axes, imgs, titles, cmaps))
            for ax, img, title, cmap in  zip_list[:5]+zip_list[8:9] + zip_list[12:13]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in zip_list[5:8]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=self.config.data.k_min-1, vmax=self.config.data.k_max)
            for ax, img, title, cmap in zip_list[9:12] + zip_list[13:]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_png_dfn(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped, gt_unwrapped_sub_wrapped_neg_norm, pred_unwrapped_sub_wrapped_neg_norm, diff_unwrapped_sub_wrapped_neg_norm = c_batch['wrapped'], c_batch['gt_unwrapped_sub_wrapped_neg_norm'], c_batch['pred_unwrapped_sub_wrapped_neg_norm'], c_batch['diff_unwrapped_sub_wrapped_neg_norm']
        # wrapped, gt_k_mat_cont, pred_k_mat_cont, diff_k_mat_cont = c_batch['wrapped'], c_batch['gt_k_mat_cont'], c_batch['pred_k_mat_cont'], c_batch['diff_k_mat_cont']
        # wrapped, gt_k_mat_disc, pred_k_mat_disc, diff_k_mat_disc = c_batch['wrapped'], c_batch['gt_k_mat_disc'], c_batch['pred_k_mat_disc'], c_batch['diff_k_mat_disc']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped", "GT unwrapped_sub_wrapped_neg_norm", "Pred unwrapped_sub_wrapped_neg_norm", "Diff unwrapped_sub_wrapped_neg_norm",
                  # "Wrapped", "GT k-mat Cont", "Pred k-mat Cont", "Diff k-mat Cont",
                  # "Wrapped", "GT k-mat Disc", "Pred k-mat Disc", "Diff k-mat Disc"
                  ]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped_sub_wrapped_neg_norm[i]), _to_numpy_2d(pred_unwrapped_sub_wrapped_neg_norm[i]), _to_numpy_2d(diff_unwrapped_sub_wrapped_neg_norm[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_cont[i]), _to_numpy_2d(pred_k_mat_cont[i]), _to_numpy_2d(diff_k_mat_cont[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_disc[i]), _to_numpy_2d(pred_k_mat_disc[i]), _to_numpy_2d(diff_k_mat_disc[i])
            ]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "inferno",
                     # "twilight", "viridis", "viridis", "inferno",
                     # "twilight", "viridis", "viridis", "inferno"
                     ]
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[:5]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[5:]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_png_neg_norm(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped_neg_norm, gt_unwrapped_neg_norm, pred_unwrapped_neg_norm, diff_unwrapped_neg_norm = c_batch['wrapped_neg_norm'], c_batch['gt_unwrapped_neg_norm'], c_batch['pred_unwrapped_neg_norm'], c_batch['diff_unwrapped_neg_norm']
        # wrapped, gt_k_mat_cont, pred_k_mat_cont, diff_k_mat_cont = c_batch['wrapped'], c_batch['gt_k_mat_cont'], c_batch['pred_k_mat_cont'], c_batch['diff_k_mat_cont']
        # wrapped, gt_k_mat_disc, pred_k_mat_disc, diff_k_mat_disc = c_batch['wrapped'], c_batch['gt_k_mat_disc'], c_batch['pred_k_mat_disc'], c_batch['diff_k_mat_disc']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped neg_norm", "GT unwrapped_neg_norm", "Pred unwrapped_neg_norm", "Pred unwrapped_neg_norm", "Diff unwrapped_neg_norm",
                  # "Wrapped", "GT k-mat Cont", "Pred k-mat Cont", "Diff k-mat Cont",
                  # "Wrapped", "GT k-mat Disc", "Pred k-mat Disc", "Diff k-mat Disc"
                  ]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped_neg_norm[i]), _to_numpy_2d(gt_unwrapped_neg_norm[i]), _to_numpy_2d(pred_unwrapped_neg_norm[i]), _to_numpy_2d(pred_unwrapped_neg_norm[i]), _to_numpy_2d(diff_unwrapped_neg_norm[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_cont[i]), _to_numpy_2d(pred_k_mat_cont[i]), _to_numpy_2d(diff_k_mat_cont[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_disc[i]), _to_numpy_2d(pred_k_mat_disc[i]), _to_numpy_2d(diff_k_mat_disc[i])
            ]
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "turbo", "inferno",
                     # "twilight", "viridis", "viridis", "inferno",
                     # "twilight", "viridis", "viridis", "inferno"
                     ]
            zip_list = list(zip(axes, imgs, titles, cmaps))
            for ax, img, title, cmap in zip_list[:2] + zip_list[3:5] + zip_list[8:9]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=0)
            for ax, img, title, cmap in zip_list[2:3]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in zip_list[5:8]+zip_list[9:]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_png_std_norm(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped_neg_norm, gt_unwrapped_std_norm, pred_unwrapped_std_norm, diff_unwrapped_std_norm = c_batch['wrapped_neg_norm'], c_batch['gt_unwrapped_std_norm'], c_batch['pred_unwrapped_std_norm'], c_batch['diff_unwrapped_std_norm']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped neg_norm", "GT unwrapped_std_norm", "Pred unwrapped_std_norm", "Diff unwrapped_std_norm",
                  ]
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped_neg_norm[i]), _to_numpy_2d(gt_unwrapped_std_norm[i]), _to_numpy_2d(pred_unwrapped_std_norm[i]), _to_numpy_2d(diff_unwrapped_std_norm[i]),
            ]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            cmaps = [
                    "twilight", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "inferno",
                     ]
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[:4]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # color_norm = colors.Normalize(vmin=-2, vmax=2)
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[4:]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_png_latent(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        wrapped, gt_unwrapped_neg_norm, pred_unwrapped_neg_norm, diff_unwrapped_neg_norm = c_batch['wrapped'], c_batch['gt_unwrapped_neg_norm'], c_batch['pred_unwrapped_neg_norm'], c_batch['diff_unwrapped_neg_norm']
        # wrapped, gt_k_mat_cont, pred_k_mat_cont, diff_k_mat_cont = c_batch['wrapped'], c_batch['gt_k_mat_cont'], c_batch['pred_k_mat_cont'], c_batch['diff_k_mat_cont']
        # wrapped, gt_k_mat_disc, pred_k_mat_disc, diff_k_mat_disc = c_batch['wrapped'], c_batch['gt_k_mat_disc'], c_batch['pred_k_mat_disc'], c_batch['diff_k_mat_disc']

        titles = ["Wrapped", "GT Unwrapped", "Pred Unwrapped", "Diff Unwrapped",
                  "Wrapped", "GT unwrapped_neg_norm", "Pred unwrapped_neg_norm", "Diff unwrapped_neg_norm",
                  # "Wrapped", "GT k-mat Cont", "Pred k-mat Cont", "Diff k-mat Cont",
                  # "Wrapped", "GT k-mat Disc", "Pred k-mat Disc", "Diff k-mat Disc"
                  ]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i]),
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped_neg_norm[i]), _to_numpy_2d(pred_unwrapped_neg_norm[i]), _to_numpy_2d(diff_unwrapped_neg_norm[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_cont[i]), _to_numpy_2d(pred_k_mat_cont[i]), _to_numpy_2d(diff_k_mat_cont[i]),
                # _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_k_mat_disc[i]), _to_numpy_2d(pred_k_mat_disc[i]), _to_numpy_2d(diff_k_mat_disc[i])
            ]
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "inferno",
                     "twilight", "turbo", "turbo", "inferno",
                     # "twilight", "viridis", "viridis", "inferno",
                     # "twilight", "viridis", "viridis", "inferno"
                     ]
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[:5]:
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            color_norm = colors.Normalize(vmin=-1, vmax=1)
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[5:]:
                im = ax.imshow(img, cmap=cmap, norm=color_norm)
                # im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _save_compare_png_ucn(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        gt_unwrapped_neg_norm, pred_unwrapped_neg_norm = c_batch['gt_unwrapped_neg_norm'],c_batch['pred_unwrapped_neg_norm']
        compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, 0)
        compare_png_path2 = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, 1)
        nrow = 4
        samples = _to_numpy_2d(gt_unwrapped_neg_norm)
        B = pred_unwrapped_neg_norm.shape[0]
        ncol = (B + nrow - 1) // nrow
        plt.figure(figsize=(nrow * 3, ncol * 3))
        for i in range(B):
            plt.subplot(ncol, nrow, i + 1)
            plt.imshow(samples[i], cmap="jet")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(compare_png_path)
        plt.close()

        samples2 = _to_numpy_2d(pred_unwrapped_neg_norm)
        plt.figure(figsize=(nrow * 3, ncol * 3))
        for i in range(B):
            plt.subplot(ncol, nrow, i + 1)
            plt.imshow(samples2[i], cmap="jet")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(compare_png_path2)
        plt.close()

    def _update_stat(self):
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")

    @cached_property
    def sampling_loader(self):
        return self.data_loader.test_loader

    @property
    def saved_samples(self):
        return self.config.io.latest_generated_sample_num

    @property
    def temp_batch_size(self):
        return min(self.config.sampling.batch_size, self.remaining_samples)

    @property
    def remaining_samples(self):
        return self.total_samples - self.saved_samples

    @cached_property
    def total_samples(self):
        return self.config.sampling.total_samples

    @cached_property
    def total_repeat_iter_num(self):
        return (self.remaining_samples // self.config.sampling.batch_size + 1) if self.remaining_samples % self.config.sampling.batch_size != 0 else self.remaining_samples // self.config.sampling.batch_size