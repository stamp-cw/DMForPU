from functools import cached_property

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
# from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

from diffusion.diffusion_setup import DiffusionSetup
from model.model_setup import ModelSetup
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
        self.eval_iter = iter(self.eval_loader)
        # with torch.no_grad(): self.ema = ExponentialMovingAverage(self.model.parameters(),
        #                                                           decay=self.config.model.ema_rate)

    def sample(self):

        if self.config.sampling.fix_seed:
            torch.manual_seed(self.config.seed)  # 设置 CPU 随机数种子
            torch.cuda.manual_seed_all(self.config.seed)  # 设置所有 GPU 随机数种子

        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        self.diffusion.setup_eval()
        # with self.ema.average_parameters():
        while self.remaining_samples > 0:
            try:
                batch_dict = next(self.eval_iter)

            except StopIteration:
                self.eval_iter = iter(self.eval_loader)
                batch_dict = next(self.eval_iter)
            self._sample(batch_dict)
            # self._save_samples_pt()
            self._save_samples_png()
            self._update_stat()

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
        return (
                    self.remaining_samples // self.config.sampling.batch_size + 1) if self.remaining_samples % self.config.sampling.batch_size != 0 else self.remaining_samples // self.config.sampling.batch_size

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
            elif self.config.diffusion.name == 'DFNDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.diff_unwrapped,
                    "gt_unwrapped_sub_wrapped_neg_norm": batch_dict["unwrapped_sub_wrapped_neg_norm"].to(self.device),
                    "pred_unwrapped_sub_wrapped_neg_norm": self.diffusion.pred_unwrapped_sub_wrapped_neg_norm,
                    "diff_unwrapped_sub_wrapped_neg_norm": batch_dict["unwrapped_sub_wrapped_neg_norm"].to(self.device) - self.diffusion.pred_unwrapped_sub_wrapped_neg_norm ,
                }
                self._save_compare_png_dfn(c_batch)
            elif self.config.diffusion.name == 'NegNormDDPMDiffusion' or self.config.diffusion.name == 'DDPMDiffusion' or self.config.diffusion.name == 'MchNegNormDDPMDiffusion' or self.config.diffusion.name == 'DphDDPMDiffusion':
                c_batch = {
                    "wrapped": self.diffusion.wrapped,
                    "gt_unwrapped": self.diffusion.gt_unwrapped,
                    "pred_unwrapped": self.diffusion.pred_unwrapped,
                    "diff_unwrapped": self.diffusion.diff_unwrapped,
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
            else:
                # c_batch = {}
                assert "NotImplementedError"

            self._save_compare_pt(c_batch)
            self.samples = self.diffusion.pred_unwrapped

    @cached_property
    def eval_loader(self):
        # return self.data_loader.eval_loader
        return self.data_loader.test_loader

    def _save_compare_pt(self, c_batch):
        pt_path = self.config.io.generated_compare_pt_file_path(self.saved_samples,
                                                               self.saved_samples + self.temp_batch_size)
        torch.save(c_batch, pt_path)
        self.logger.info(f"Saved {self.temp_batch_size} samples to {pt_path}")


    # def _save_samples_pt(self):
    #     self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
    #     self.samples = self.samples.reshape(
    #         (-1, self.config.data.image_size, self.config.data.image_size, self.config.data.color_channels))
    #     pt_path = self.config.io.generated_sample_pt_file_path(self.saved_samples,
    #                                                            self.saved_samples + self.temp_batch_size)
    #     torch.save(self.samples, pt_path)
    #     self.logger.info(f"Saved {self.temp_batch_size} samples to {pt_path}")

    def _save_samples_png(self):
        # self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        # self.samples = self.samples.reshape(
        #     (-1, self.config.data.image_size, self.config.data.image_size, self.config.data.color_channels))
        # samples = self.samples.permute(0, 3, 1, 2)
        # samples = self.samples
        # for _, img_array in enumerate(samples):
        #     img = ToPILImage()(img_array)
        #     img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1)
        #     img.save(img_path)

        samples = self.samples  # shape: [B, H, W] 或 [B, 1, H, W]
        for i, img_array in enumerate(samples):
            img = img_array.squeeze()  # [H, W]
            img = img.detach().cpu().numpy()
            img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
            plt.figure(figsize=(4, 4))
            plt.imshow(img, cmap="turbo")   # ⭐ 改这里的颜色
            plt.axis("off")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
            plt.close()
        self.logger.info(
            f"Saved {self.samples.shape[0]} samples as PNG images in folder: {self.config.io.out_raw_sample_path}")

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
        loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
        # self.ema.load_state_dict(loaded_state['ema'])
        self.diffusion.model.load_state_dict(loaded_state['model'])
        if self.config.diffusion.use_controlnet:
            self.diffusion.controlnet_model.load_state_dict(loaded_state['controlnet_model'])

    def data_inverse_scaler(self, x):
        from selector.data_selector import BaseDataLoader
        data_loader = BaseDataLoader(self.config)
        return data_loader.data_inverse_scaler(x)

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
    # def _save_samples_and_preview(self, raw_images, label_images, mask_images):
    #     self.samples = mask_images
    #     self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
    #     samples_to_visualize = self.samples[:min(36, len(self.samples))]
    #     # visualize_samples_file_path = self.config.io.sample_pdf_file_path(0)
    #     visualize_samples_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
    #                                                                                 self.saved_samples + self.temp_batch_size,
    #                                                                                 0)
    #     samples_grid_format = samples_to_visualize.permute(0, 3, 1, 2)
    #     grid_tensor = make_grid(samples_grid_format, nrow=int(len(samples_to_visualize) ** 0.5))
    #     grid_image = ToPILImage()(grid_tensor.cpu())
    #     grid_image.save(visualize_samples_file_path, format='PDF')
    #
    #     new_raw_images = torch.clamp(raw_images.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
    #     raw_to_visualize = new_raw_images[:min(36, len(self.samples))]
    #     # visualize_raw_file_path = self.config.io.sample_pdf_file_path(1)
    #     visualize_raw_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
    #                                                                             self.saved_samples + self.temp_batch_size,
    #                                                                             1)
    #     raw_grid_format = raw_to_visualize.permute(0, 3, 1, 2)
    #     grid_tensor = make_grid(raw_grid_format, nrow=int(len(raw_to_visualize) ** 0.5))
    #     grid_image = ToPILImage()(grid_tensor.cpu())
    #     grid_image.save(visualize_raw_file_path, format='PDF')
    #
    #     new_label_images = torch.clamp(label_images.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
    #     label_to_visualize = new_label_images[:min(36, len(self.samples))]
    #     # visualize_label_file_path = self.config.io.sample_pdf_file_path(2)
    #     visualize_label_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
    #                                                                               self.saved_samples + self.temp_batch_size,
    #                                                                               2)
    #     label_grid_format = label_to_visualize.permute(0, 3, 1, 2)
    #     grid_tensor = make_grid(label_grid_format, nrow=int(len(label_to_visualize) ** 0.5))
    #     grid_image = ToPILImage()(grid_tensor.cpu())
    #     grid_image.save(visualize_label_file_path, format='PDF')
    #
    #     mask_images = torch.clamp(mask_images.cpu() * 255, 0, 255).to(torch.uint8)
    #     assert raw_images.shape[0] == label_images.shape[0] == mask_images.shape[0]
    #     # visualize_merged_file_path = self.config.io.sample_pdf_file_path(3)
    #     visualize_merged_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
    #                                                                                self.saved_samples + self.temp_batch_size,
    #                                                                                3)
    #     images_list = []
    #     for i in range(raw_images.shape[0]):
    #         raw = ToPILImage()(raw_images[i].cpu())
    #         label = ToPILImage()(label_images[i].cpu())
    #         mask = ToPILImage()(mask_images[i].cpu())
    #         # 横向拼接
    #         width, height = raw.width + label.width + mask.width, raw.height
    #         merged = Image.new('RGB', (width, height))
    #         merged.paste(raw, (0, 0))
    #         merged.paste(label, (raw.width, 0))
    #         merged.paste(mask, (raw.width + label.width, 0))
    #         images_list.append(merged)
    #
    #     # 保存为PDF
    #     if images_list:
    #         images_list[0].save(visualize_merged_file_path, save_all=True, append_images=images_list[1:], format='PDF')

    def _update_stat(self):
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        # self.iter_num += 1