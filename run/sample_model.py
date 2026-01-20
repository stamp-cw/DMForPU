import os

import torch
from functools import cached_property

from model.mmodel_setup import MModelSetup
# from model.model_setup import ModelSetup
from selector.data_selector import _DATA_LOADERS
import matplotlib.pyplot as plt
from matplotlib import colors

class ModelSampler:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = config.test.device
        self.mmodel = MModelSetup(self.config, self.logger).mmodel
        self.data_loader = _DATA_LOADERS(self.config)
        self.test_iter = iter(self.eval_loader)

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
        loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
        self.mmodel.model.load_state_dict(loaded_state['model'], strict=True)

    # def load_checkpoint(self):
    #     # self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
    #     # loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
    #     # self.mmodel.model.load_state_dict(loaded_state['model'], strict=True)
    #     load_weights = r"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Big/model_weights/weights.pth"
    #     checkpoint = torch.load(load_weights, map_location='cpu')
    #
    #     ckpt_file_path = os.path.join(self.config.io.out_ckpt_path, f'{self.config.io.out_ckpt_filename_prefix}_{100}.pth')
    #     # self.vae.model.save_pretrained(self.config.io.out_hf_ckpt_path)
    #     state_dict = {
    #         'model': checkpoint['state_dict'],
    #         'optimizer': checkpoint['optimizer'],
    #         'epoch': 100
    #     }
    #     torch.save(state_dict, ckpt_file_path)
    #
    #     self.mmodel.model.load_state_dict(checkpoint['state_dict'])

    def sample(self):
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        self.mmodel.setup_eval()
        while self.remaining_samples > 0:
            try:
                batch_dict = next(self.test_iter)
            except StopIteration:
                self.test_iter = iter(self.eval_loader)
                batch_dict = next(self.test_iter)
            self._sample(batch_dict)

    def _sample(self, batch_dict):
        with torch.no_grad():
            # self.diffusion.infer_sample()
            wrapped = batch_dict["wrapped"].to(self.device)
            pred_unwrapped = self.mmodel.eval_predict(wrapped)
            # self._save_samples_and_preview(wrapped, gt_unwrapped, pred_unwrapped)
            c_batch = {
                "wrapped": batch_dict["wrapped"].to(self.device),
                "gt_unwrapped": batch_dict["unwrapped"].to(self.device),
                "pred_unwrapped": pred_unwrapped,
                "diff_unwrapped": batch_dict["unwrapped"].to(self.device) - pred_unwrapped,
                # "gt_unwrapped_neg_norm": batch_dict["unwrapped_neg_norm"].to(self.device),
            }
            self._save_compare_png(c_batch)
            self._save_compare_pt(c_batch)
            self.samples = self.mmodel.pred
        self._save_samples_png()
        self._update_stat()

    @cached_property
    def eval_loader(self):
        # return self.data_loader.eval_loader
        return self.data_loader.test_loader

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

    def _save_compare_png(self, c_batch):
        def _to_numpy_2d(x: torch.Tensor):
            return x.detach().cpu().squeeze().numpy()
        wrapped, gt_unwrapped, pred_unwrapped, diff_unwrapped = c_batch['wrapped'], c_batch['gt_unwrapped'], c_batch['pred_unwrapped'], c_batch['diff_unwrapped']
        titles = ["Wrapped", "GT Unwrapped", "Pred unwrapped", "Diff Unwrapped"]
        # color_norm = colors.Normalize(vmin=-1, vmax=16)
        for i in range(wrapped.shape[0]):
            compare_png_path = self.config.io.generated_compare_png_file_path(self.saved_samples,self.saved_samples + self.temp_batch_size, i)
            imgs = [
                _to_numpy_2d(wrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(gt_unwrapped[i]), _to_numpy_2d(pred_unwrapped[i]), _to_numpy_2d(diff_unwrapped[i])
            ]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes = axes.flatten()
            cmaps = ["twilight", "turbo", "turbo", "inferno"]
            for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps)):
                # im = ax.imshow(img, cmap=cmap, norm=color_norm)
                im = ax.imshow(img, cmap=cmap)
                ax.set_title(title)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # color_norm = colors.Normalize(vmin=-1, vmax=1)
            # for ax, img, title, cmap in list(zip(axes, imgs, titles, cmaps))[2:]:
            #     im = ax.imshow(img, cmap=cmap, norm=color_norm)
            #     # im = ax.imshow(img, cmap=cmap)
            #     ax.set_title(title)
            #     ax.axis("off")
            #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(compare_png_path, dpi=200)
            plt.close(fig)

    def _update_stat(self):
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        # self.iter_num += 1

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
