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
from run.losses import LossFN
from model.optimizer import OptimizerFN
from selector.data_selector import _DATA_LOADERS
from selector.optimizer_selector import _OPTIMIZERS
from utils.metrics import l1_metric, rmse_metric
from utils.util import AverageMeter
from torch.cuda.amp import autocast, GradScaler

METRIC_FUNCS = {
    "l1": l1_metric,
    "rmse": rmse_metric,
}

class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.device = self.config.training.device
        self.diffusion = DiffusionSetup(self.config, self.logger).diffusion
        self.optimizer = _OPTIMIZERS(self.config)(self.diffusion.optimize_parameters)
        self.data_loader = _DATA_LOADERS(self.config)
        # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.config.model.ema_rate)
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
            self.diffusion.setup_train()
            avg_meter = AverageMeter()
            # avg_eval_meter = AverageMeter()
            pbar = tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.end_epoch}")
            # for batch_data in self.train_loader:
            for batch_data in pbar:
                self.acc_batch += 1
                metrics_dict = self.epoch_fn(self.diffusion, self.optimizer, epoch, batch_data)
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
            'model': self.diffusion.model.state_dict(),
            # 'ema': self.ema.state_dict(),
            'control_model': self.diffusion.control_model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, ckpt_file_path)
        self.logger.info(f"Saved model to {ckpt_file_path}")
        if self.config.io.use_wandb and self.config.io.save_pth_to_wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(ckpt_file_path)
            wandb.log_artifact(artifact)

        if self.config.training.snapshot_sampling: self._snapshot_sampling()

    def _load_state(self):
        ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
        self.diffusion.model.load_state_dict(ckpt['model'])
        # self.ema.load_state_dict(ckpt['ema'])
        self.diffusion.control_model.load_state_dict(ckpt['control_model'])
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

    # def _record_and_evaluate_fold(self):
    #     self.saved = False
    #     if self.epoch % self.config.training.eval_freq == 0:
    #         self.evaluate_loss = 0
    #         self.model.eval()
    #         avg_meter = AverageMeter()
    #         for batch_data in self.eval_loader:
    #             batch_data = [self.data_loader.data_scaler(x).to(self.device) if hasattr(x, "to") else x for x in
    #                           batch_data]
    #             with torch.no_grad():
    #                 metrics = self.eval_epoch_fn(self.model, self.optimizer, self.ema, self.epoch, batch_data)
    #                 loss = metrics["loss"]
    #                 avg_meter.update(metrics)
    #                 self.evaluate_loss += loss
    #
    #         avg_metrics = avg_meter.avg()
    #         self._record_metrics(avg_metrics, f"eval_epoch_fold_{self.config.data.fold}", self.epoch)
    #         self.eval_fold_metrics = avg_metrics
    #         self.model.train()

    # def _evaluate(self):
    #     self.saved = False
    #     if self.epoch % self.config.training.eval_freq == 0:
    #         self.evaluate_loss = 0
    #         self.model.eval()
    #         avg_meter = AverageMeter()
    #         for batch_data in self.eval_loader:
    #             # batch_data = batch_data.to(self.device)
    #             # batch_data = self.data_loader.data_scaler(batch_data)
    #             batch_data = [self.data_loader.data_scaler(x).to(self.device) if hasattr(x, "to") else x for x in batch_data]
    #             with torch.no_grad():
    #                 # loss = self.eval_epoch_fn(self.model, self.optimizer, self.ema, self.epoch, batch_data)
    #                 # self.evaluate_loss += loss.item()
    #                 metrics = self.eval_epoch_fn(self.model, self.optimizer, self.ema, self.epoch, batch_data)
    #                 loss = metrics["loss"]
    #                 avg_meter.update(metrics)
    #                 self.evaluate_loss += loss
    #         avg_metrics = avg_meter.avg()
    #         self._record_metrics(avg_metrics, "eval_epoch", self.epoch)
    #         self.evaluate_loss /= len(self.eval_loader)
    #         if self.config.io.use_tensorboard: self.writer.add_scalar("evaluate_loss", self.evaluate_loss, self.epoch)
    #         self._update_best_evaluate()
    #         self.model.train()

    # def _update_best_evaluate(self):
    #     if self.epoch - self.start_epoch > self.config.training.eval_save_least_epoch and self.evaluate_loss < self.best_evaluate_loss:
    #         self._save_state(self.epoch)
    #         self.saved = True
    #     self.best_evaluate_loss = min(self.best_evaluate_loss, self.evaluate_loss)
    #     self.logger.info(
    #         f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Eval Loss: {self.evaluate_loss:.4f}, Best Eval Loss: {self.best_evaluate_loss:.4f}")

    @property
    def train_loader(self):
        return self.data_loader.train_loader if not self.config.training.use_all_data else self.data_loader.all_loader

    @property
    def eval_loader(self):
        return self.data_loader.eval_loader

    def _snapshot_sampling(self):
        from run.sample import Sampler
        self.config.sampling.batch_size = self.config.training.snapshot_batch_size
        self.config.sampling.total_samples = self.config.training.snapshot_batch_size
        self.config.sampling.eval = True
        sampler = Sampler(self.config)
        # sampler.ema = self.ema
        # sampler.model = self.model
        sampler.diffusion = self.diffusion
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

    def epoch_fn(self, diffusion, optimizer, epoch, batch):
        # wrapped, gt_unwrapped = batch
        diffusion.setup_data(batch)
        t_batch = torch.randint(0, self.config.diffusion.num_train_timesteps, (1,), device=self.config.training.device).long().expand(batch.shape[0])
        diffusion.train_sample(t_batch)
        pred_unwrapped = diffusion.pred_unwrapped
        gt_unwrapped = diffusion.gt_unwrapped
        # list(diffusion.model.parameters()) + list(diffusion.control_model.parameters())

        if self.train:
            if self.config.training.amp:
                scaler = GradScaler()
                optimizer.zero_grad()
                with autocast():
                    loss = self.loss_fn(diffusion)
                scaler.scale(loss).backward()
                self.optimize_fn(optimizer, diffusion.optimize_parameters, epoch=epoch, scaler=scaler)
                self.metrics["loss"] = loss.item()

            else:
                optimizer.zero_grad()
                loss = self.loss_fn(diffusion)
                loss.backward()
                self.optimize_fn(optimizer, diffusion.optimize_parameters, epoch=epoch)
                self.metrics["loss"] = loss.item()
        else:
            loss = self.loss_fn(diffusion)
            self.metrics["loss"] = loss.item()

        for k in self.metrics_keys:
            func = METRIC_FUNCS[k]
            value = func(pred_unwrapped, gt_unwrapped)
            self.metrics[k] = value.item()

        return self.metrics

