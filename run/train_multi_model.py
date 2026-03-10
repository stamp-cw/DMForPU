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

from meter.meter_setup import MeterSetup
from model.mmodel_setup import MModelSetup
from run.losses import LossFN
from model.optimizer import OptimizerFN
from selector.data_selector import _DATA_LOADERS
from selector.optimizer_selector import _OPTIMIZERS
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        if self.config.model.name == "U3Net":
            from accelerate import DistributedDataParallelKwargs
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=True
            )
            self.accelerator = Accelerator(
                kwargs_handlers=[ddp_kwargs]
            )
        else:
            self.accelerator = Accelerator()
        config.accelerator = self.accelerator
        self.logger = config.logger
        self.device = self.config.training.device
        self.mmodel = MModelSetup(self.config, self.logger).mmodel
        self.optimizer = _OPTIMIZERS(self.config)(self.mmodel.optimize_parameters)
        self.data_loader = _DATA_LOADERS(self.config)
        self.optimize_fn = OptimizerFN(self.config)
        self.best_evaluate_loss = self.config.training.best_evaluate_loss
        if self.config.io.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.config.io.tensorboard_path)
            config.writer = self.writer
        self.meter = MeterSetup(self.config, self.logger).meter
        self.main_meter = MeterSetup(self.config, self.logger).meter
        config.train_meter = self.meter
        self.meter.mode = 'train_multi_model'
        self.meter.is_record = False
        self.main_meter.mode = 'train_multi_model'
        self.main_meter.is_record = True
        self.epoch_fn = EpochFN(optimize_fn=self.optimize_fn, config=self.config)


    def train(self):
        # if self.config.accelerator.is_main_process:
        if self.config.io.training_from_scratch or self.config.io.latest_checkpoint_file_path is None:
            self.start_epoch = 0
            self.acc_batch = 0
            self.end_epoch = self.config.training.brand_new_epochs
        elif not self.config.io.training_from_scratch and self.config.io.latest_checkpoint_file_path is not None:
            self.start_epoch = self.config.io.latest_checkpoint_epoch + 1
            self.acc_batch = self.start_epoch * len(self.train_loader)
            self.end_epoch = self.start_epoch + self.config.training.continue_training_epochs
            if self.accelerator.is_main_process:
                self.logger.info(f"Continuing training from epoch {self.start_epoch}")
            self._load_state()
        self._train()

    def _train(self):
        self.mmodel.model, self.optimizer, self.m_train_loader = self.accelerator.prepare(
            self.mmodel.model, self.optimizer, self.train_loader
        )
        for epoch in range(self.start_epoch, self.end_epoch):
            # self.optimizer.zero_grad(set_to_none=True)
            self.epoch = epoch
            self.mmodel.setup_train()
            pbar = tqdm.tqdm(self.m_train_loader,
                             total=len(self.m_train_loader),
                             disable=not self.accelerator.is_main_process,
                             desc=f"Epoch {epoch}/{self.end_epoch}")
            for step, batch_data in enumerate(pbar):
                self.acc_batch += 1
                self.meter.acc_step = self.acc_batch
                if self.accelerator.is_main_process:
                    self.main_meter.acc_step = self.acc_batch
                self.epoch_fn(self.accelerator, self.mmodel, self.optimizer, self.meter, self.main_meter, epoch, batch_data)
                self.meter.epoch_meter.update(self.meter.batch_metric_dict)
            if self.accelerator.is_main_process:
                self.main_meter.compute_epoch_metric()
                self._record_and_evaluate()
            # self.meter.compute_epoch_metric()
            # # loss_tensor = torch.tensor(
            # #     self.meter.epoch_metric_dict["loss"],
            # #     device=self.accelerator.device
            # # )
            # # loss_all = self.accelerator.gather(loss_tensor)
            # avg_metrics = self.gather_metrics(
            #     self.accelerator,
            #     self.meter.epoch_metric_dict
            # )
            # if self.config.accelerator.is_main_process:
            #     # self.avg_loss= self.meter.epoch_metric_dict['loss']
            #     self.avg_loss= avg_metrics['loss']
            #     # self.avg_loss = loss_all.mean().item()
            #     self._record_and_evaluate()
            # # torch.cuda.empty_cache()

    # def gather_metrics(self, accelerator, metrics: dict):
    #     out = {}
    #     for k, v in metrics.items():
    #         t = torch.tensor(v, device=accelerator.device)
    #         # 官方推荐用 reduce
    #         out[k] = accelerator.reduce(t, reduction="mean").item()
    #     return out

    def _save_state(self, epoch):
        ckpt_file_path = os.path.join(self.config.io.out_ckpt_path, f'{self.config.io.out_ckpt_filename_prefix}_{epoch}.pth')
        state_dict = {
            'model': self.mmodel.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        # torch.save(state_dict, ckpt_file_path)
        self.config.accelerator.save(state_dict, ckpt_file_path)
        self.logger.info(f"Saved model to {ckpt_file_path}")
        if self.config.io.use_wandb and self.config.io.save_pth_to_wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(ckpt_file_path)
            wandb.log_artifact(artifact)

        # if self.config.training.snapshot_sampling: self._snapshot_sampling()
        # if self.config.training.snapshot_val: self._snapshot_val(epoch)

    def _load_state(self):
        ckpt = torch.load(self.config.io.latest_checkpoint_file_path, map_location=self.device, weights_only=False)
        self.mmodel.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

    def _record_and_evaluate(self):
        # # if self.config.io.use_tensorboard: self.writer.add_scalar("training_loss", self.avg_loss, self.epoch)
        # if self.epoch % self.config.training.log_freq == 0:
        #     # self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.avg_loss:.4f}")
        #     self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.main_meter.epoch_metric_dict}")
        # if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch - 1 and not self.saved and self.epoch != 0:
        #     self._save_state(self.epoch)
        if self.epoch % self.config.training.log_freq == 0:
            self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Loss: {self.main_meter.epoch_metric_dict}")
        if self.epoch % self.config.training.snapshot_freq == 0 or self.epoch == self.end_epoch - 1 and self.epoch != 0:
            self._save_state(self.epoch)
        if self.config.training.snapshot_val and self.epoch % self.config.training.snapshot_val_freq == 0 or self.epoch == self.end_epoch - 1 and self.epoch != 0:
            self._snapshot_val(self.epoch)
        if self.config.training.snapshot_sampling and self.epoch % self.config.training.snapshot_sampling_freq == 0 or self.epoch == self.end_epoch - 1 and self.epoch != 0:
            self._snapshot_sampling(self.epoch)

    @property
    def train_loader(self):
        return self.data_loader.train_loader if not self.config.training.use_all_data else self.data_loader.all_loader

    @property
    def val_loader(self):
        return self.data_loader.test_loader

    @property
    def sampling_loader(self):
        return self.data_loader.test_loader

    def _snapshot_sampling(self, epoch=0):
        from run.sample_model import ModelSampler as Sampler
        self.config.sampling.batch_size = self.config.training.snapshot_batch_size
        self.config.sampling.total_samples = self.config.training.snapshot_batch_size
        self.config.sampling_from_epoch = epoch
        sampler = Sampler(self.config)
        sampler.sampling_loader = self.sampling_loader
        sampler.mmodel = self.mmodel
        sampler.sample()

    def _snapshot_val(self, epoch):
        from run.val_model import ModelValidator as Valuator
        self.config.val.batch_size = self.config.training.snapshot_batch_size
        # self.config.iio.use_wandb = False
        valuator = Valuator(self.config)
        valuator.save_pt = False
        valuator.epoch = epoch
        valuator.meter = self.main_meter
        valuator.meter.writer = self.writer if self.config.io.use_tensorboard else None
        valuator.meter.mode = 'val_model'
        valuator.mmodel = self.mmodel
        valuator.val_loader = self.val_loader
        valuator.valuate()
        # save 最好的val_metric
        if self.config.training.snapshot_best_loss:
            self.evaluate_loss = self.main_meter.epoch_metric_dict['NRMSE']
            self._update_best_evaluate()
        self.meter.mode = 'train_multi_model'
        self.main_meter.mode = 'train_multi_model'

    def _update_best_evaluate(self):
        if self.evaluate_loss < self.best_evaluate_loss:
            self._save_state(self.epoch)
        self.best_evaluate_loss = min(self.best_evaluate_loss, self.evaluate_loss)
        self.logger.info(f"Epoch {self.epoch}/{self.end_epoch - self.start_epoch}, Eval Loss: {self.evaluate_loss:.4f}, Best Eval Loss: {self.best_evaluate_loss:.4f}")

class EpochFN:
    def __init__(self, optimize_fn, config):
        self.optimize_fn = optimize_fn
        self.config = config
        self.loss_fn = LossFN(self.config)

    def __call__(self, accelerator, mmodel, optimizer, meter, main_meter, epoch, batch):
        return self.epoch_fn(accelerator, mmodel, optimizer, meter, main_meter, epoch, batch)

    def epoch_fn(self, accelerator, mmodel, optimizer, meter, main_meter, epoch, batch):
        mmodel.setup_data(batch)
        mmodel.train_predict(batch)
        pred_batch = mmodel.pred_batch
        if accelerator.is_main_process:
            main_meter.epoch = epoch
        meter.epoch = epoch
        meter.setup_data(pred_batch)
        meter.compute_batch_metric()
        with accelerator.accumulate(mmodel.model):
            # optimizer.zero_grad()
            loss = self.loss_fn(mmodel)
            # loss.backward()
            accelerator.backward(loss)
            # self.optimize_fn(optimizer, mmodel.optimize_parameters, epoch=epoch)
            if accelerator.sync_gradients:
                self.optimize_fn(optimizer, mmodel.optimize_parameters, epoch=epoch)
                optimizer.zero_grad(set_to_none=True)
        # loss_all = self.config.accelerator.gather(loss.detach())
        # loss_mean = loss_all.mean()
        # meter.batch_metric_dict["loss"] = loss_mean.item()
        meter.batch_metric_dict["loss"] = loss
        batch_metric_dict = meter.batch_metric_dict
        for k, v in batch_metric_dict.items():
            if isinstance(v, torch.Tensor):
                batch_metric_dict[k] = v.detach()
        m_acc_batch_metric_dict = accelerator.gather_for_metrics(batch_metric_dict)
        # print(f"b:{batch_metric_dict},rank={accelerator.process_index},local_rank={accelerator.local_process_index}, device={accelerator.device}")
        # print(f"m_acc:{m_acc_batch_metric_dict},rank={accelerator.process_index},local_rank={accelerator.local_process_index}, device={accelerator.device}")
        if accelerator.is_main_process:
            # print("m_acc_batch_metric_dict:", m_acc_batch_metric_dict)
            m_reduce_batch_metric_dict = {
                k: v.mean().item() if isinstance(v, torch.Tensor) else v
                for k, v in  m_acc_batch_metric_dict.items()
            }
            # print("m_reduce_batch_metric_dict:", m_reduce_batch_metric_dict)
            main_meter._record_metrics(m_reduce_batch_metric_dict, f"{main_meter.mode}_per_batch", main_meter.acc_step)
            main_meter.epoch_meter.update(m_reduce_batch_metric_dict)
        # return meter


