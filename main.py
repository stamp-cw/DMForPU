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
from utils.util import dict2namespace, unflatten_dict, update_dict


def main():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the configs file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'val', 'test','train_vae', 'test_vae','train_model','test_model','val_model','sample_model'], help='Train the model or generate samples')
    parser.add_argument('--user_logging_level', type=str, required=False, default='info', choices=['debug', 'info', 'warning', 'error'], help='Set logging level (debug, info, warning, error)')
    parser.add_argument('--training_from_scratch', action='store_true', default=False, required=False, help='Train from scratch instead of resuming training')
    parser.add_argument('--sampling_from_epoch', type=int, required=False, default=None, help='Epoch number to load for sampling (default: latest checkpoint)')
    parser.add_argument('--hyper', required=False, action='store_true',default=False, help='hyper train')
    parser.add_argument('--debug', required=False, action='store_true',default=False, help='debug')

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
            if args.debug:
                from run.train_t import Trainer
                trainer = Trainer(config)
                trainer.train()
            else:
                from run.train import Trainer
                trainer = Trainer(config)
                trainer.train()
        elif args.mode == 'sample':
            if args.debug:
                from run.sample_t import Sampler
                sampler = Sampler(config)
                sampler.load_checkpoint()
                sampler.sample()
            else:
                from run.sample import Sampler
                sampler = Sampler(config)
                sampler.load_checkpoint()
                sampler.sample()
        elif args.mode == 'val':
            from run.val import Valuator
            valuator = Valuator(config)
            valuator.load_checkpoint()
            valuator.valuate()
        elif args.mode == 'test':
            from run.test import Tester
            tester = Tester(config)
            tester.load_checkpoint()
            tester.test()
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




if __name__ == '__main__':
    # Import Optimizers
    from model.optimizer import AdamOptimizer,SGDOptimizer
    # Import Losses
    from run.losses import PureLossType, Pure2LossType
    from run.losses import GEWLOSSType
    from run.losses import PHY1LossType,PHY2LossType,PHY3LossType, PHY4LossType
    from run.losses import PHY11LossType, PHY12LossType, PHY13LossType
    from run.losses import VAEKLLossType, VAEPURELossType
    from run.losses import UNetLOSSLossType
    # Import VAE Models
    from vae.latent_vae import LatentVAE
    # Import Models
    from model.unet.aux_unet import AuxUNet
    from model.unet_mmodel import UNetMModel
    from model.aux_unet_mmodel import AuxUNetMModel
    from model.diff_aux_unet_mmodel import DiffAuxUNetMModel
    from model.unet.diff_aux_unet import DiffAuxUNet
    from model.pu_unet_mmodel import PuUNetMModel
    from model.unet.pu_unet import PuUNet
    # Import Diffusions
    from diffusion.ddpm_diffusion import DDPMDiffusion
    from diffusion.neg_norm_ddpm_diffusion import NegNormDDPMDiffusion
    from diffusion.mch_neg_norm_ddpm_diffusion import MchNegNormDDPMDiffusion
    from diffusion.dfn_ddpm_diffusion import  DFNDDPMDiffusion
    from diffusion.kdf_ddpm_diffusion import  KdfDDPMDiffusion
    from diffusion.latent_ddpm_diffusion import LatentDDPMDiffusion
    from diffusion.dph_ddpm_diffusion import DphDDPMDiffusion
    from diffusion.elucidate_diffusion import ElucidatedDiffusion
    from diffusion.ucn_ddpm_diffusion import UcnDDPMDiffusion
    from diffusion.phase_ddpm_diffusion import PhaseDDPMDiffusion
    from diffusion.phase_cut_ddpm_diffusion import PhaseCutDDPMDiffusion
    # Import Meter
    from meter.pure_meter import PureMeter
    main()