# -*- coding: utf-8 -*-
"""
@File    : train_vae.py
@Author  : 18744
@Time    : 2026/1/09 10:51
@Description :
"""
import os
import tqdm
import torch
import wandb
from run.losses import LossFN
from model.optimizer import OptimizerFN
from selector.data_selector import _DATA_LOADERS
from selector.optimizer_selector import _OPTIMIZERS
from utils.metrics import l1_metric
from utils.util import AverageMeter
from torch.cuda.amp import autocast, GradScaler

from vae.vae_setup import VAESetup

METRIC_FUNCS = {
    "l1": l1_metric,
}

class VAETrainer:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = self.config.training.device
        self.vae = VAESetup(self.config, self.logger).model
        self.optimizer = _OPTIMIZERS(self.config)(self.vae.optimize_parameters)
        self.data_loader = _DATA_LOADERS(self.config)
        self.optimize_fn = OptimizerFN(self.config)
        self.epoch_fn = EpochFN(train=True, optimize_fn=self.optimize_fn, config=self.config)
        # self.eval_epoch_fn = EpochFN(train=False, optimize_fn=self.optimize_fn, config=self.config)
        # self.best_evaluate_loss = self.config.training.best_evaluate_loss
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.io.tensorboard_path)

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
            self.vae.setup_train()
            avg_meter = AverageMeter()
            # avg_eval_meter = AverageMeter()
            pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.end_epoch}")
            # for batch_data in self.train_loader:
            for step, batch_data in enumerate(pbar):
                self.acc_batch += 1
                metrics_dict = self.epoch_fn(self.vae, self.optimizer, epoch, batch_data)
                self._record_metrics(metrics_dict, "train_per_batch", self.acc_batch)
                avg_meter.update(metrics_dict)
            avg_metrics = avg_meter.avg()
            # avg_eval_metrics = avg_eval_meter.avg()
            self._record_metrics(avg_metrics, "train_per_epoch", self.epoch)
            # self._record_metrics(avg_eval_metrics, "eval_per_epoch", self.epoch)
            self.avg_loss = avg_metrics["loss"]
            self._record_and_evaluate()

    def _save_state(self, epoch):
        ckpt_file_path = os.path.join(self.config.io.out_ckpt_path, f'{self.config.io.out_ckpt_filename_prefix}_{epoch}.pth')
        state_dict = {
            'model': self.vae.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, ckpt_file_path)
        self.logger.info(f"Saved model to {ckpt_file_path}")
        if self.config.io.use_wandb and self.config.io.save_pth_to_wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(ckpt_file_path)
            wandb.log_artifact(artifact)

        # if self.config.training.snapshot_sampling: self._snapshot_sampling()

    def _load_state(self):
        ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
        self.vae.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def _record_metrics(self, metrics, prefix, step):
        if self.config.io.use_tensorboard:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def _record_and_evaluate(self):
        if self.config.io.use_tensorboard: self.writer.add_scalar("training_loss", self.avg_loss, self.epoch)
        if self.epoch % self.config.training.log_freq == 0: self.logger.info(
            f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.avg_loss:.4f}")
        # self._evaluate()
        if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch - 1 and not self.saved and self.epoch != 0:
            self._save_state(self.epoch)

    @property
    def train_loader(self):
        return self.data_loader.train_loader if not self.config.training.use_all_data else self.data_loader.all_loader

    @property
    def eval_loader(self):
        return self.data_loader.eval_loader

    def _snapshot_sampling(self):
        from run.test_vae import VAETester as Tester
        self.config.sampling.batch_size = self.config.training.snapshot_batch_size
        self.config.sampling.total_samples = self.config.training.snapshot_batch_size
        self.config.sampling.eval = True
        sampler = Tester(self.config)
        # sampler.ema = self.ema
        # sampler.model = self.model
        sampler.vae = self.vae
        sampler.sample()


class EpochFN:
    def __init__(self, train, optimize_fn, config):
        self.train = train
        self.optimize_fn = optimize_fn
        self.config = config
        self.loss_fn = LossFN(self.config)
        metrics_dict = vars(self.config.metrics)
        self.metrics = {k: [] for k, v in metrics_dict.items() if v}
        self.metrics_keys = list(self.metrics.keys())

    def __call__(self, diffusion, optimizer, epoch, batch):
        return self.epoch_fn(diffusion, optimizer, epoch, batch)

    def epoch_fn(self, vae, optimizer, epoch, batch):
        gt = batch['unwrapped_neg_norm']
        pred = vae.predict(gt)

        if self.train:
            if self.config.training.amp:
                scaler = GradScaler()
                optimizer.zero_grad()
                with autocast():
                    loss = self.loss_fn(vae)
                scaler.scale(loss).backward()
                self.optimize_fn(optimizer, vae.optimize_parameters, epoch=epoch, scaler=scaler)
                self.metrics["loss"] = loss.item()
            else:
                optimizer.zero_grad()
                loss = self.loss_fn(vae)
                loss.backward()
                self.optimize_fn(optimizer, vae.optimize_parameters, epoch=epoch)
                self.metrics["loss"] = loss.item()
        else:
            loss = self.loss_fn(vae)
            self.metrics["loss"] = loss.item()

        for k in self.metrics_keys:
            func = METRIC_FUNCS[k]
            value = func(pred, gt)
            self.metrics[k] = value.item()

        return self.metrics

