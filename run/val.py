# -*- coding: utf-8 -*-
"""
@File    : val.py
@Author  : 18744
@Time    : 2025/10/3 18:19
@Description :
"""
from functools import cached_property

import torch
import tqdm

from diffusion.diffusion_setup import DiffusionSetup
from meter.meter_setup import MeterSetup
from selector.data_selector import _DATA_LOADERS


class Valuator:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.epoch = config.sampling_from_epoch
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"{self.config.io.tensorboard_path}/{config.mode}")
            config.writer = self.writer
        self.meter = MeterSetup(self.config, self.logger).meter
        self.device = config.val.device
        self.data_loader = _DATA_LOADERS(self.config)
        self.diffusion = DiffusionSetup(self.config, self.logger).diffusion

    # def load_checkpoint(self):
    #     self.logger.info(f"Loading checkpoint from {self.config.io.val_ckpt_file_path}")
    #     loaded_state = torch.load(self.config.io.val_ckpt_file_path, map_location=self.device, weights_only=True)
    #     self.diffusion.model.load_state_dict(loaded_state['model'])

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.val_ckpt_file_path}")
        state_dict = torch.load(self.config.io.val_ckpt_file_path, map_location=self.device, weights_only=True)['model']
        is_multi_card = any(k.startswith("module.") for k in state_dict.keys())
        if is_multi_card:
            self.logger.info("Detected multi-card checkpoint. Stripping 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            self.logger.info("Detected single-card checkpoint.")
        self.diffusion.model.load_state_dict(state_dict)


    def valuate(self):
        self.diffusion.setup_eval()
        # print(len(self.val_loader))
        self.meter.acc_step = self.epoch * len(self.val_loader)
        for batch_dict in tqdm.tqdm(self.val_loader, desc=f"Epoch {self.epoch} Valuating"):
            self._valuate(batch_dict)
        self.meter.compute_epoch_metric()
        self._save_val_pt()
        self._update_stat()

    def _valuate(self, batch_dict):
        with torch.no_grad():
            self.diffusion.setup_data(batch_dict)
            # wrapped = batch_dict["wrapped"].to(self.device)
            self.diffusion.infer_sample()
            pred_batch = self.diffusion.pred_batch
            self.meter.epoch = self.epoch
            self.meter.setup_data(pred_batch)
            self.meter.acc_step += 1
            self.meter.compute_batch_metric()
            self.meter.epoch_meter.update(self.meter.batch_metric_dict)

    def _save_val_pt(self):
        pt_path = self.config.io.generated_val_pt_file_path(self.epoch)
        torch.save(self.meter.epoch_metric_dict, pt_path)
        self.logger.info(f"Saved Epoch {self.epoch} val result to {pt_path}")

    def _update_stat(self):
        # self.logger.info(f"Total samples to generate: {self.total_samples}; Already generated {self.saved_samples}; Remaining: {self.remaining_samples}")
        # print(self.meter.epoch_metric_dict)
        # self.logger.info(f"Epoch {self.epoch}, L1: {self.meter.epoch_metric_dict['L1']:.4f}, RMSE: {self.meter.epoch_metric_dict['RMSE']:.4f}")
        self.logger.info(f"Epoch {self.epoch}, indicate: {self.meter.epoch_metric_dict}")

    @cached_property
    def val_loader(self):
        return self.data_loader.test_loader