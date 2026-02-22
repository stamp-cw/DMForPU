import shutil
from functools import cached_property

import torch
import tqdm

from model.mmodel_setup import MModelSetup
from meter.meter_setup import MeterSetup
from selector.data_selector import _DATA_LOADERS

class ModelValidator:
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
        self.mmodel = MModelSetup(self.config, self.logger).mmodel
        self.save_pt = False

    def load_checkpoint(self):
        self.logger.info(f"Loading checkpoint from {self.config.io.val_ckpt_file_path}")
        state_dict = torch.load(self.config.io.val_ckpt_file_path, map_location=self.device, weights_only=True)['model']
        is_multi_card = any(k.startswith("module.") for k in state_dict.keys())
        if is_multi_card:
            self.logger.info("Detected multi-card checkpoint. Stripping 'module.' prefix...")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        else:
            self.logger.info("Detected single-card checkpoint.")
        self.mmodel.model.load_state_dict(state_dict)

    def valuate(self):
        self.mmodel.setup_eval()
        # print(len(self.val_loader))
        self.meter.acc_step = self.epoch * len(self.val_loader)
        for step, batch_dict in enumerate(tqdm.tqdm(self.val_loader, desc=f"Epoch {self.epoch} Valuating")):
            self._valuate(batch_dict)
            if self.save_pt:
                self._save_val_batch_pt(self.save_batch, step)
        self.meter.compute_epoch_metric()
        if self.save_pt:
            self._save_val_pt()
        self._update_stat()

    def _valuate(self, batch_dict):
        with torch.no_grad():
            self.mmodel.setup_data(batch_dict)
            self.mmodel.eval_predict(batch_dict)
            pred_batch = self.mmodel.pred_batch
            self.meter.epoch = self.epoch
            self.meter.setup_data(pred_batch)
            self.meter.acc_step += 1
            self.meter.compute_batch_metric()
            self.meter.epoch_meter.update(self.meter.batch_metric_dict)
            self.save_batch = self.meter.batch_metric_dict
            if getattr(self.config.val, "save_raw_batch_pt", False):
                self.save_batch.update(pred_batch)
            # print(self.save_batch)

    def _save_val_batch_pt(self, c_batch, batch_idx):
        pt_path = self.config.io.generated_val_batch_pt_file_path(self.config.val.batch_size, batch_idx, batch_idx * self.config.val.batch_size, (batch_idx + 1) * self.config.val.batch_size)
        torch.save(c_batch, pt_path)
        self.logger.info(f"Saved batch {batch_idx} to {pt_path}")

    def _save_val_pt(self):
        pt_path = self.config.io.generated_val_pt_file_path(self.epoch)
        torch.save(self.meter.epoch_metric_dict, pt_path)
        self.logger.info(f"Saved Epoch {self.epoch} val result to {pt_path}")

    def _update_stat(self):
        # with open(f"{self.config.io.out_val_path}/val_epoch_metric.txt", 'w', encoding='utf-8') as f:
        #     f.write(str(self.meter.epoch_metric_dict))
        self.logger.info(f"Epoch {self.epoch}, indicate: {self.meter.epoch_metric_dict}")
        if self.save_pt:
            shutil.copy(self.config.logger.logger_file_path, self.config.io.out_val_path)

    @cached_property
    def val_loader(self):
        return self.data_loader.test_loader