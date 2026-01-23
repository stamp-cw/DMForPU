# -*- coding: utf-8 -*-
"""
@File    : main.py
@Author  : 18744
@Time    : 2025/1/14 10:21
@Description :
"""
import argparse
import os
import sys
import yaml
from utils.logger import Logger
import wandb
from datetime import datetime
from configs.dynamic_io import IOConfig

def main():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the configs file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'eval', 'train_vae', 'test_vae','train_model','test_model','val_model','sample_model'], help='Train the model or generate samples')
    parser.add_argument('--user_logging_level', type=str, required=False, default='info', choices=['debug', 'info', 'warning', 'error'], help='Set logging level (debug, info, warning, error)')
    parser.add_argument('--training_from_scratch', action='store_true', default=False, required=False, help='Train from scratch instead of resuming training')
    parser.add_argument('--sampling_from_epoch', type=int, required=False, default=None, help='Epoch number to load for sampling (default: latest checkpoint)')
    parser.add_argument('--hyper', required=False, action='store_true',default=False, help='hyper train')

    args = parser.parse_args()

    logger = Logger(args.user_logging_level)
    logger.info(f"Config file: {args}")

    # Fusion yaml file and command line arguments
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)

    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tmp_config = dict2namespace(config)
    tmp_config.iio.in_dataset_path = os.path.join("data", tmp_config.data.name)
    tmp_config.iio.in_dataset_stat_path = os.path.join("data", f"{tmp_config.data.name}.npz")
    tmp_config.iio.in_raw_dataset_path = os.path.join("data", "raw", tmp_config.data.name)
    if args.hyper:
        tmp_config.iio.out_asset_suffix = os.path.join("hyper",tmp_config.data.name, tmp_config.model.name, f"exp-{run_time}")
    elif args.mode == 'train_vae' or args.mode == 'test_vae':
        tmp_config.iio.out_asset_suffix = os.path.join(tmp_config.data.name, tmp_config.model.name)
    elif args.mode == 'train_model' or args.mode == 'test_model' or args.mode == 'val_model' or args.mode == 'sample_model':
        tmp_config.iio.out_asset_suffix = os.path.join(tmp_config.data.name, tmp_config.model.name)
    else:
        tmp_config.iio.out_asset_suffix = os.path.join(tmp_config.data.name, tmp_config.diffusion.name)


    ioo = IOConfig(tmp_config)
    ioo.user_logging_level = args.user_logging_level
    ioo.training_from_scratch = args.training_from_scratch
    ioo.sampling_from_epoch = args.sampling_from_epoch

    if ioo.use_wandb and ioo.use_tensorboard:
        # Initialize wandb
        if not wandb.api.api_key:
            wandb.login()

        if args.hyper:
            with open(os.path.join('configs', 'hyper_config.yaml'), 'r') as f:
                hyper_config = yaml.safe_load(f)
            wandb.init(
                project=tmp_config.model.name,
                name=f"exp-{tmp_config.data.name}-{run_time}",
                config=hyper_config,
                sync_tensorboard=True,
                dir=ioo.wandb_local_path,
                resume=not ioo.training_from_scratch
            )
            wandb_cfg_unflat = unflatten_dict(dict(wandb.config))
            config = update_dict(config, wandb_cfg_unflat)
        else:
            wandb.init(
                project=tmp_config.model.name,
                name=f"exp-{tmp_config.data.name}-{run_time}",
                config=config,
                sync_tensorboard=True,
                dir = ioo.wandb_local_path,
                resume= not ioo.training_from_scratch
            )
            config = wandb.config

    config = dict2namespace(config)
    config.mode = args.mode
    config.io = ioo

    # Print and log the Current configuration
    logger.debug("Current configuration:")
    for key, value in config.__dict__.items():
        log_message = f"{key}: {value}"
        logger.debug(log_message)

    config.logger = logger

    # Train or sample
    try:
        if args.mode == 'train':
            from run.train import Trainer
            trainer = Trainer(config)
            trainer.train()
        elif args.mode == 'sample':
            from run.sample import Sampler
            sampler = Sampler(config)
            sampler.load_checkpoint()
            sampler.sample()
        elif args.mode == 'eval':
            from run.eval import Evaluator
            evaluator = Evaluator(config)
            evaluator.load_checkpoint()
            evaluator.evaluate()
        elif args.mode == 'train_vae':
            from run.train_vae import VAETrainer
            vae_trainer = VAETrainer(config)
            vae_trainer.train()
        elif args.mode == 'test_vae':
            from run.test_vae import VAETester
            vae_tester = VAETester(config)
            vae_tester.load_checkpoint()
            vae_tester.test()
        elif args.mode  == 'train_model':
            from run.train_model import ModelTrainer
            model_trainer = ModelTrainer(config)
            model_trainer.train()
        elif args.mode  == 'test_model':
            from run.test_model import ModelTester
            model_tester = ModelTester(config)
            model_tester.load_checkpoint()
            model_tester.test()
        elif args.mode  == 'val_model':
            from run.val_model import ModelValidator
            model_validator = ModelValidator(config)
            model_validator.load_checkpoint()
            model_validator.validate()
        elif args.mode  == 'sample_model':
            from run.sample_model import ModelSampler
            model_sampler = ModelSampler(config)
            model_sampler.load_checkpoint()
            model_sampler.sample()
        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    if config.io.use_wandb:
        wandb.finish()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            # 自动尝试把字符串转成数字
            if isinstance(value, str):
                try:
                    if value.lower() in ["true", "false"]:   # 转成 bool
                        new_value = value.lower() == "true"
                    elif "." in value or "e" in value.lower():  # float
                        new_value = float(value)
                    else:  # int
                        new_value = int(value)
                except Exception:
                    new_value = value  # 转换失败就保持字符串
            else:
                new_value = value

        setattr(namespace, key, new_value)
    return namespace

def update_dict(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            update_dict(base[k], v)
        else:
            base[k] = v
    return base

def unflatten_dict(flat_dict, sep="."):
    result = {}
    for key, value in flat_dict.items():
        parts = key.split(sep)
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    return result

if __name__ == '__main__':
    from model.optimizer import AdamOptimizer,SGDOptimizer
    from run.losses import PHY1LossType,PHY2LossType,PHY3LossType,PureLossType, PHY4LossType, PHY11LossType, PHY12LossType
    from diffusion.ddpm_diffusion import DDPMDiffusion
    from diffusion.neg_norm_ddpm_diffusion import NegNormDDPMDiffusion
    from diffusion.mch_neg_norm_ddpm_diffusion import MchNegNormDDPMDiffusion
    from diffusion.dfn_ddpm_diffusion import  DFNDDPMDiffusion
    from diffusion.kdf_ddpm_diffusion import  KdfDDPMDiffusion
    from diffusion.latent_ddpm_diffusion import LatentDDPMDiffusion
    from diffusion.dph_ddpm_diffusion import DphDDPMDiffusion
    from vae.latent_vae import LatentVAE
    from run.losses import VAEKLLossType, VAEPURELossType, UNetLOSSLossType
    # from model.unet.unet_model import UNet
    # from model.unet.unet import UNet as AuxUNet
    # from model.unet.unet import UNet
    from model.unet.aux_unet import AuxUNet
    from model.unet_mmodel import UNetMModel
    from model.aux_unet_mmodel import AuxUNetMModel
    from model.diff_aux_unet_mmodel import DiffAuxUNetMModel
    from model.unet.diff_aux_unet import DiffAuxUNet
    from model.pu_unet_mmodel import PuUNetMModel
    from model.unet.pu_unet import PuUNet
    from diffusion.elucidate_diffusion import ElucidatedDiffusion
    from diffusion.ucn_ddpm_diffusion import UcnDDPMDiffusion
    from diffusion.phase_ddpm_diffusion import PhaseDDPMDiffusion
    main()