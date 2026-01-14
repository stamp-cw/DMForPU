from functools import cached_property

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch_ema import ExponentialMovingAverage
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from Model.model_setup import ModelSetup
# from run.sde import ScoreFN
from selector.data_selector import _DATA_LOADERS


class Sampler:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = config.sampling.device
        self.model = ModelSetup(self.config, self.logger).model
        self.data_loader = _DATA_LOADERS(self.config)
        self.eval_iter = iter(self.eval_loader)
        # self.score_fn = ScoreFN()
        with torch.no_grad(): self.ema = ExponentialMovingAverage(self.model.parameters(),
                                                                  decay=self.config.model.ema_rate)

    def sample(self):
        # self.iter_num = 0
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        self.model.eval()
        with self.ema.average_parameters():
            while self.remaining_samples > 0:
                # images, labels = next(iter(self.eval_loader))
                try:
                    images, labels = next(self.eval_iter)
                except StopIteration:
                    self.eval_iter = iter(self.eval_loader)
                    images, labels = next(self.eval_iter)
                self._sample(images, labels)
                self._save_samples_pt()
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

    def _sample(self, images, labels):
        with torch.no_grad():
            output = self.model(images)
            mask = torch.sigmoid(output)
            mask = self.data_inverse_scaler(mask) > self.config.sampling.out_threshold
            self._save_samples_and_preview(images, labels, mask)
            self.samples = mask

    @cached_property
    def eval_loader(self):
        # return self.data_loader.eval_loader
        return self.data_loader.test_loader

    def _save_samples_pt(self):
        self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        self.samples = self.samples.reshape(
            (-1, self.config.data.image_size, self.config.data.image_size, self.config.data.color_channels))
        pt_path = self.config.io.generated_sample_pt_file_path(self.saved_samples,
                                                               self.saved_samples + self.temp_batch_size)
        torch.save(self.samples, pt_path)
        self.logger.info(f"Saved {self.temp_batch_size} samples to {pt_path}")

    def _save_samples_png(self):
        samples = self.samples.permute(0, 3, 1, 2)
        for _, img_array in enumerate(samples):
            img = ToPILImage()(img_array)
            img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1)
            img.save(img_path)
        self.logger.info(
            f"Saved {self.samples.shape[0]} samples as PNG images in folder: {self.config.io.out_raw_sample_path}")

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
        loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
        self.ema.load_state_dict(loaded_state['ema'])
        self.model.load_state_dict(loaded_state['model'], strict=True)

    def data_inverse_scaler(self, x):
        from selector.data_selector import BaseDataLoader
        data_loader = BaseDataLoader(self.config)
        return data_loader.data_inverse_scaler(x)

    def _save_samples_and_preview(self, raw_images, label_images, mask_images):
        self.samples = mask_images
        self.samples = torch.clamp(self.samples.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        samples_to_visualize = self.samples[:min(36, len(self.samples))]
        # visualize_samples_file_path = self.config.io.sample_pdf_file_path(0)
        visualize_samples_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
                                                                                    self.saved_samples + self.temp_batch_size,
                                                                                    0)
        samples_grid_format = samples_to_visualize.permute(0, 3, 1, 2)
        grid_tensor = make_grid(samples_grid_format, nrow=int(len(samples_to_visualize) ** 0.5))
        grid_image = ToPILImage()(grid_tensor.cpu())
        grid_image.save(visualize_samples_file_path, format='PDF')

        new_raw_images = torch.clamp(raw_images.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        raw_to_visualize = new_raw_images[:min(36, len(self.samples))]
        # visualize_raw_file_path = self.config.io.sample_pdf_file_path(1)
        visualize_raw_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
                                                                                self.saved_samples + self.temp_batch_size,
                                                                                1)
        raw_grid_format = raw_to_visualize.permute(0, 3, 1, 2)
        grid_tensor = make_grid(raw_grid_format, nrow=int(len(raw_to_visualize) ** 0.5))
        grid_image = ToPILImage()(grid_tensor.cpu())
        grid_image.save(visualize_raw_file_path, format='PDF')

        new_label_images = torch.clamp(label_images.permute(0, 2, 3, 1).cpu() * 255, 0, 255).to(torch.uint8)
        label_to_visualize = new_label_images[:min(36, len(self.samples))]
        # visualize_label_file_path = self.config.io.sample_pdf_file_path(2)
        visualize_label_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
                                                                                  self.saved_samples + self.temp_batch_size,
                                                                                  2)
        label_grid_format = label_to_visualize.permute(0, 3, 1, 2)
        grid_tensor = make_grid(label_grid_format, nrow=int(len(label_to_visualize) ** 0.5))
        grid_image = ToPILImage()(grid_tensor.cpu())
        grid_image.save(visualize_label_file_path, format='PDF')

        mask_images = torch.clamp(mask_images.cpu() * 255, 0, 255).to(torch.uint8)
        assert raw_images.shape[0] == label_images.shape[0] == mask_images.shape[0]
        # visualize_merged_file_path = self.config.io.sample_pdf_file_path(3)
        visualize_merged_file_path = self.config.io.generated_sample_pdf_file_path(self.saved_samples,
                                                                                   self.saved_samples + self.temp_batch_size,
                                                                                   3)
        images_list = []
        for i in range(raw_images.shape[0]):
            raw = ToPILImage()(raw_images[i].cpu())
            label = ToPILImage()(label_images[i].cpu())
            mask = ToPILImage()(mask_images[i].cpu())
            # 横向拼接
            width, height = raw.width + label.width + mask.width, raw.height
            merged = Image.new('RGB', (width, height))
            merged.paste(raw, (0, 0))
            merged.paste(label, (raw.width, 0))
            merged.paste(mask, (raw.width + label.width, 0))
            images_list.append(merged)

        # 保存为PDF
        if images_list:
            images_list[0].save(visualize_merged_file_path, save_all=True, append_images=images_list[1:], format='PDF')

    def _update_stat(self):
        self.logger.info(
            f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        # self.iter_num += 1