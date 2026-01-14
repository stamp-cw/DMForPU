import abc
import os
import glob
import re
import ml_collections
from functools import cached_property


class DynamicIOConfig(ml_collections.ConfigDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_asset_prefix(self):
        return "assets"

    @property
    def out_sample_suffix(self):
        return "sample"

    @property
    def out_eval_suffix(self):
        return "eval"

    @property
    def out_stat_suffix(self):
        return "stat"

    @property
    def out_sample_raw_suffix(self):
        return "raw"

    @property
    def out_sample_filename_prefix(self):
        return 'sample'

    @property
    def out_eval_filename_prefix(self):
        return 'eval'

    @property
    def out_ckpt_suffix(self):
        return 'ckpt'

    @property
    def out_ckpt_filename_prefix(self):
        return 'epoch'

    @property
    def tensorboard_path_suffix(self):
        return 'tb'

    @property
    def wandb_local_path_suffix(self):
        return 'wandb'

    @property
    @abc.abstractmethod
    def in_dataset_path(self):
        pass

    @property
    @abc.abstractmethod
    def in_dataset_stat_path(self):
        pass

    @property
    @abc.abstractmethod
    def in_raw_dataset_path(self):
        pass

    @property
    @abc.abstractmethod
    def out_asset_suffix(self):
        pass

    @property
    @abc.abstractmethod
    def use_tensorboard(self):
        pass

    @property
    @abc.abstractmethod
    def use_wandb(self):
        pass

    @property
    @abc.abstractmethod
    def save_pth_to_wandb(self):
        pass

    @cached_property
    def out_ckpt_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_ckpt_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def out_sample_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_sample_suffix,
                            str(self.sampling_epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def out_eval_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_eval_suffix,
                            str(self.sampling_epoch))
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def out_raw_sample_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_sample_suffix,
                            str(self.sampling_epoch), self.out_sample_raw_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def out_raw_eval_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_eval_suffix,
                            str(self.sampling_epoch), self.out_sample_raw_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def out_stat_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.out_stat_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def out_fid_stat_path(self):
        path = os.path.join(self.out_stat_path, 'fid_stats.npz')
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    @cached_property
    def out_fid_png_path(self):
        path = os.path.join(self.out_stat_path, 'fid_png')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def tensorboard_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.tensorboard_path_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def wandb_local_path(self):
        path = os.path.join(self.out_asset_prefix, self.out_asset_suffix, self.wandb_local_path_suffix)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def get_epoch_num(filename):
        match = re.search(r'_(\d+)\.pth', filename)
        return int(match.group(1)) if match else 0

    @property
    def latest_checkpoint_epoch(self):
        pattern = os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_*.pth')
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            return None
        latest_ckpt_file = max(ckpt_files, key=self.get_epoch_num)
        epoch_num = self.get_epoch_num(latest_ckpt_file)
        return epoch_num

    @property
    def latest_checkpoint_file_path(self):
        if self.latest_checkpoint_epoch is None:
            return None
        return os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_{self.latest_checkpoint_epoch}.pth')

    @property
    def sampling_epoch(self):
        if self.sampling_from_epoch is not None:
            return self.sampling_from_epoch
        else:
            return self.latest_checkpoint_epoch

    @property
    def sampling_ckpt_file_path(self):
        if self.sampling_from_epoch is not None:
            return os.path.join(self.out_ckpt_path, f'{self.out_ckpt_filename_prefix}_{self.sampling_from_epoch}.pth')
        else:
            return self.latest_checkpoint_file_path

    def generated_sample_pt_file_path(self, start_image_count, end_image_count):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_sample_path,
                            f"{self.out_sample_filename_prefix}s_{start_image_count}_{end_image_count}.pt")

    def generated_eval_pt_file_path(self, start_image_count, end_image_count):
        if self.latest_generated_eval_num is None:
            return None
        return os.path.join(self.out_eval_path,
                            f"{self.out_eval_filename_prefix}s_{start_image_count}_{end_image_count}.pt")

    def generated_sample_png_file_path(self, image_count):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_raw_sample_path, f"{self.out_sample_filename_prefix}_{image_count:05d}.png")

    def generated_eval_png_file_path(self, image_count):
        if self.latest_generated_eval_num is None:
            return None
        return os.path.join(self.out_raw_eval_path, f"{self.out_eval_filename_prefix}_{image_count:05d}.png")

    def sample_pdf_file_path(self, step):
        return os.path.join(self.out_sample_path, f"{self.out_sample_filename_prefix}_{step}.pdf")

    def generated_sample_pdf_file_path(self, start_image_count, end_image_count, step):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_sample_path,
                            f"{self.out_sample_filename_prefix}s_{start_image_count}_{end_image_count}_{step}.pdf")

    def generated_eval_pdf_file_path(self, start_image_count, end_image_count, step):
        if self.latest_generated_eval_num is None:
            return None
        return os.path.join(self.out_eval_path,
                            f"{self.out_eval_filename_prefix}s_{start_image_count}_{end_image_count}_{step}.pdf")

    @staticmethod
    def get_sample_num(filename):
        match = re.search(r'_(\d+)\.png', filename)
        return int(match.group(1)) if match else 0

    @property
    def latest_generated_sample_num(self):
        pattern = os.path.join(self.out_raw_sample_path, f'{self.out_sample_filename_prefix}_*.png')
        sample_files = glob.glob(pattern)
        if not sample_files:
            return 0
        latest_sample_file = max(sample_files, key=self.get_sample_num)
        latest_sample_num = self.get_sample_num(latest_sample_file)
        return latest_sample_num

    @property
    def latest_generated_eval_num(self):
        pattern = os.path.join(self.out_raw_eval_path, f'{self.out_eval_filename_prefix}_*.png')
        sample_files = glob.glob(pattern)
        if not sample_files:
            return 0
        latest_sample_file = max(sample_files, key=self.get_sample_num)
        latest_sample_num = self.get_sample_num(latest_sample_file)
        return latest_sample_num

    @property
    def latest_generated_sample_file_path(self):
        if self.latest_generated_sample_num is None:
            return None
        return os.path.join(self.out_raw_sample_path,
                            f'{self.out_sample_filename_prefix}_{self.latest_generated_sample_num}.png')


class IOConfig(DynamicIOConfig):

    def __init__(self, config):
        super().__init__()
        self._in_dataset_path = config.iio.in_dataset_path
        self._in_dataset_stat_path = config.iio.in_dataset_stat_path
        self._in_raw_dataset_path = config.iio.in_raw_dataset_path
        self._out_asset_suffix = config.iio.out_asset_suffix
        self._use_tensorboard = config.iio.use_tensorboard
        self._use_wandb = config.iio.use_wandb
        self._save_pth_to_wandb = config.iio.save_pth_to_wandb

    @property
    def in_dataset_path(self): return self._in_dataset_path

    # def in_dataset_path(self): return os.path.join("data", "SARBuD")

    @property
    def in_dataset_stat_path(self): return self._in_dataset_stat_path

    # def in_dataset_stat_path(self): return os.path.join("data", "SARBuD.npz")

    @property
    def in_raw_dataset_path(self): return self._in_raw_dataset_path

    # def in_raw_dataset_path(self): return os.path.join("data", "raw", "SARBuD")

    @property
    def out_asset_suffix(self): return self._out_asset_suffix

    # def out_asset_suffix(self): return os.path.join("ve", "SARBuD_deponet_cont")

    @property
    def use_tensorboard(self): return self._use_tensorboard

    # def use_tensorboard(self): return True

    @property
    def use_wandb(self): return self._use_wandb

    # def use_wandb(self): return False

    @property
    def save_pth_to_wandb(self): return self._save_pth_to_wandb

    # def save_pth_to_wandb(self): return False

    @in_dataset_path.setter
    def in_dataset_path(self, value):
        self._in_dataset_path = value

    @in_dataset_stat_path.setter
    def in_dataset_stat_path(self, value):
        self._in_dataset_stat_path = value

    @in_raw_dataset_path.setter
    def in_raw_dataset_path(self, value):
        self._in_raw_dataset_path = value

    @out_asset_suffix.setter
    def out_asset_suffix(self, value):
        self._out_asset_suffix = value

    @use_tensorboard.setter
    def use_tensorboard(self, value):
        self._use_tensorboard = value

    @use_wandb.setter
    def use_wandb(self, value):
        self._use_wandb = value

    @save_pth_to_wandb.setter
    def save_pth_to_wandb(self, value):
        self._save_pth_to_wandb = value