# -*- coding: utf-8 -*-
"""
@File    : train.py
@Author  : 18744
@Time    : 2025/11/14 10:51
@Description :
"""
import os
import tqdm
import torch
import wandb
from diffusion.diffusion_setup import DiffusionSetup
from meter.meter_setup import MeterSetup
from run.losses import LossFN
from model.optimizer import OptimizerFN
from selector.data_selector import _DATA_LOADERS
from selector.optimizer_selector import _OPTIMIZERS
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = self.config.training.device
        self.diffusion = DiffusionSetup(self.config, self.logger).diffusion
        self.optimizer = _OPTIMIZERS(self.config)(self.diffusion.optimize_parameters)
        self.data_loader = _DATA_LOADERS(self.config)
        self.optimize_fn = OptimizerFN(self.config)
        self.meter = MeterSetup(self.config, self.logger).meter
        self.epoch_fn = EpochFN(optimize_fn=self.optimize_fn, config=self.config)
        config.train_meter = self.meter
        self.meter.mode = 'train'
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.io.tensorboard_path)
            config.writer = self.writer

    def train(self):
        if self.config.io.training_from_scratch or self.config.io.latest_checkpoint_file_path is None:
            self.start_epoch = 0
            self.acc_batch = 0
            self.end_epoch = self.config.training.brand_new_epochs
        elif not self.config.io.training_from_scratch and self.config.io.latest_checkpoint_file_path is not None:
            self.start_epoch = self.config.io.latest_checkpoint_epoch
            self.acc_batch = self.start_epoch * len(self.train_loader)
            self.end_epoch = self.start_epoch + self.config.training.continue_training_epochs
            self.logger.info(f"Continuing training from epoch {self.start_epoch}")
            self._load_state()
        self._train()

    def _train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.diffusion.setup_train()
            pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.end_epoch}")
            for batch_data in pbar:
                self.acc_batch += 1
                self.config.acc_step = self.acc_batch
                self.epoch_fn(self.diffusion, self.optimizer, epoch, batch_data)
            self.meter.compute_epoch_metric()
            self._record_and_evaluate()

    def _save_state(self, epoch):
        ckpt_file_path = os.path.join(self.config.io.out_ckpt_path, f'{self.config.io.out_ckpt_filename_prefix}_{epoch}.pth')
        if self.config.diffusion.use_controlnet:
            state_dict = {
                'model': self.diffusion.model.state_dict(),
                # 'ema': self.ema.state_dict(),
                'controlnet_model': self.diffusion.controlnet_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                "epoch": epoch
            }
        else:
            state_dict = {
                'model': self.diffusion.model.state_dict(),
                # 'ema': self.ema.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch
            }
        torch.save(state_dict, ckpt_file_path)
        self.logger.info(f"Saved model to {ckpt_file_path}")
        if self.config.io.use_wandb and self.config.io.save_pth_to_wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(ckpt_file_path)
            wandb.log_artifact(artifact)

        if self.config.training.snapshot_sampling: self._snapshot_sampling()
        if self.config.training.snapshot_val: self._snapshot_val(epoch)

    def _load_state(self):
        ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
        self.diffusion.model.load_state_dict(ckpt['model'])
        # self.ema.load_state_dict(ckpt['ema'])
        if self.config.diffusion.use_controlnet:
            self.diffusion.controlnet_model.load_state_dict(ckpt['controlnet_model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def _record_and_evaluate(self):
        if self.epoch % self.config.training.log_freq == 0: self.logger.info(
            f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.avg_loss:.4f}")
        if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch - 1 and not self.saved and self.epoch != 0:
            self._save_state(self.epoch)

    @property
    def train_loader(self):
        return self.data_loader.train_loader if not self.config.training.use_all_data else self.data_loader.all_loader

    @property
    def val_loader(self):
        return self.data_loader.eval_loader

    @property
    def sampling_loader(self):
        return self.data_loader.eval_loader

    def _snapshot_sampling(self):
        from run.sample import Sampler
        self.config.sampling.batch_size = self.config.training.snapshot_batch_size
        self.config.sampling.total_samples = self.config.training.snapshot_batch_size
        # self.config.sampling.eval = True
        sampler = Sampler(self.config)
        sampler.sampling_loader = self.sampling_loader
        sampler.diffusion = self.diffusion
        sampler.sample()

    def _snapshot_val(self, epoch):
        from run.val import Valuator
        self.config.val.batch_size = self.config.training.snapshot_batch_size
        valuator = Valuator(self.config)
        valuator.epoch = epoch
        valuator.meter = self.meter
        valuator.meter.mode = 'val'
        valuator.diffusion = self.diffusion
        valuator.val_loader = self.val_loader
        valuator.valuate()
        self.meter.mode = 'train'

class EpochFN:
    def __init__(self, optimize_fn, config):
        self.optimize_fn = optimize_fn
        self.config = config
        self.loss_fn = LossFN(self.config)
        self.meter = config.meter

    def __call__(self, diffusion, optimizer, epoch, batch):
        return self.epoch_fn(diffusion, optimizer, epoch, batch)

    def epoch_fn(self, diffusion, optimizer, epoch, batch):
        diffusion.setup_data(batch)
        t_batch = torch.randint(0, self.config.diffusion.num_train_timesteps, (1,), device=self.config.training.device).long().expand(self.config.training.batch_size)
        diffusion.train_sample(t_batch)
        pred_batch = diffusion.pred_batch

        self.meter.epoch = epoch
        self.meter.setup_batch_data(pred_batch)
        self.meter.compute_batch_metric()
        if self.config.training.amp:
            scaler = GradScaler()
            optimizer.zero_grad()
            with autocast():
                loss = self.loss_fn(diffusion)
            scaler.scale(loss).backward()
            self.optimize_fn(optimizer, diffusion.optimize_parameters, epoch=epoch, scaler=scaler)
        else:
            optimizer.zero_grad()
            loss = self.loss_fn(diffusion)
            loss.backward()
            self.optimize_fn(optimizer, diffusion.optimize_parameters, epoch=epoch)
        self.meter.batch_metric_dict["loss"] = loss.item()