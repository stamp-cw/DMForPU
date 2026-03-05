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
from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the configs file')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'val', 'test','train_vae', 'test_vae','train_model','test_model','val_model','sample_model','train_multi_model','train_multi'], help='Train the model or generate samples')
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
    elif args.mode == 'train_model' or args.mode == 'test_model' or args.mode == 'val_model' or args.mode == 'sample_model' or args.mode == 'train_multi_model':
        tmp_config.iio.out_asset_suffix = os.path.join(tmp_config.data.name, tmp_config.model.name)
    else:
        tmp_config.iio.out_asset_suffix = os.path.join(tmp_config.data.name, tmp_config.diffusion.name)

    if args.mode == 'train_multi':
        accelerator = Accelerator(split_batches=True)

    ioo = IOConfig(tmp_config)
    ioo.user_logging_level = args.user_logging_level
    ioo.training_from_scratch = args.training_from_scratch
    ioo.sampling_from_epoch = args.sampling_from_epoch

    # if args.mode == 'train_multi':
    #     ioo.use_wandb = False

    if ioo.use_wandb and ioo.use_tensorboard:
        # Initialize wandb
        if not wandb.api.api_key:
            wandb.login()

        if args.hyper:
            with open(os.path.join('configs', 'hyper_config.yaml'), 'r') as f:
                hyper_config = yaml.safe_load(f)
            # wandb.tensorboard.patch()
            wandb.init(
                project=tmp_config.model.name,
                name=f"exp-{tmp_config.data.name}-{run_time}",
                config=hyper_config,
                # sync_tensorboard=True if accelerator.is_main_process else False,
                sync_tensorboard=False,
                dir=ioo.wandb_local_path,
                resume=not ioo.training_from_scratch
            )
            wandb_cfg_unflat = unflatten_dict(dict(wandb.config))
            config = update_dict(config, wandb_cfg_unflat)
        else:
            # ioo.use_wandb = False
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
    if args.hyper:
        config.loss_type.bleta1 = wandb.config.loss_bleta1
        config.loss_type.bleta2 = wandb.config.loss_bleta2
        config.loss_type.bleta3 = wandb.config.loss_bleta3
        if not accelerator.is_main_process:
            ioo.use_wandb = False
    config.mode = args.mode
    config.sampling_from_epoch = args.sampling_from_epoch
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
        elif args.mode == 'train_multi':
            from run.train_multi import Trainer as MultiTrainer
            if config.io.use_tensorboard:
                from torch.utils.tensorboard import SummaryWriter
                config.writer = SummaryWriter(config.io.tensorboard_path)
            config.accelerator = accelerator
            multi_trainer = MultiTrainer(config)
            multi_trainer.train()
        elif args.mode == 'sample':
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
        elif args.mode  == 'train_multi_model':
            from run.train_multi_model import ModelTrainer as MultiModelTrainer
            multi_model_trainer = MultiModelTrainer(config)
            multi_model_trainer.train()
        elif args.mode  == 'test_model':
            from run.test_model import ModelTester
            model_tester = ModelTester(config)
            model_tester.load_checkpoint()
            model_tester.test()
        elif args.mode  == 'val_model':
            from run.val_model import ModelValidator
            model_validator = ModelValidator(config)
            model_validator.load_checkpoint()
            model_validator.valuate()
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
    # from run.losses import GEWLOSSType
    from run.losses import PHY1LossType,PHY2LossType,PHY3LossType, PHY4LossType
    from run.losses import PHY11LossType, PHY12LossType, PHY13LossType
    from run.losses import VAEKLLossType, VAEPURELossType
    from run.losses import UNetLOSSLossType
    from run.losses import MchGradLossType
    from run.losses import SqdLstmLossType
    from run.losses import PUNetLossType
    from run.losses import RestormerLossType
    from run.losses import UformerLossType
    from run.losses import WavLossType
    from run.losses import DLPULossType
    from run.losses import U3NetLossType
    # Import VAE Models
    from vae.latent_vae import LatentVAE
    # Import Models
    from model.unet.aux_unet import AuxUNet
    from model.u3net.u3net import U3Net
    from model.aux_unet_mmodel import AuxUNetMModel
    from model.diff_aux_unet_mmodel import DiffAuxUNetMModel
    from model.unet.diff_aux_unet import DiffAuxUNet
    from model.punet_mmodel import PUNetMModel
    from model.unet.punet import PUNet
    from model.lstm.sqd_lstm import JointConvSQDLSTMNet
    from model.transformer.restormer import Restormer
    from model.transformer.uformer import Uformer
    from model.unet.dlpu import DLPUNet
    from model.fdunet.fdunet import FDUNet
    # Import Diffusions
    # from diffusion.ddpm_diffusion import DDPMDiffusion
    # from diffusion.neg_norm_ddpm_diffusion import NegNormDDPMDiffusion
    # from diffusion.mch_neg_norm_ddpm_diffusion import MchNegNormDDPMDiffusion
    # from diffusion.dfn_ddpm_diffusion import  DFNDDPMDiffusion
    # from diffusion.kdf_ddpm_diffusion import  KdfDDPMDiffusion
    # from diffusion.latent_ddpm_diffusion import LatentDDPMDiffusion
    # from diffusion.dph_ddpm_diffusion import DphDDPMDiffusion
    # from diffusion.elucidate_diffusion import ElucidatedDiffusion
    # from diffusion.ucn_ddpm_diffusion import UcnDDPMDiffusion
    # from diffusion.phase_ddpm_diffusion import PhaseDDPMDiffusion
    # from diffusion.phase_cut_ddpm_diffusion import PhaseCutDDPMDiffusion
    from diffusion.mch_ddpm_diffusion import MchDDPMDiffusion
    # from diffusion.grad_ddpm_diffusion import GradDDPMDiffusion
    from diffusion.wav_ddpm_diffusion import WavDDPMDiffusion
    from diffusion.cfg_ddpm_diffusion import CfgDDPMDiffusion
    from diffusion.dfn_ddpm_diffusion import DfnDDPMDiffusion
    # from diffusion.fdu_ddpm_diffusion import FduDDPMDiffusion
    from diffusion.fdu_ddpm_diffusion_v1 import FduDDPMDiffusion
    # from diffusion.fdu_ddpm_diffusion_v3 import FduDDPMDiffusion
    # from diffusion.old.fdu_ddpm_diffusion import FduDDPMDiffusion
    # Import Meter
    from meter.pure_meter import PureMeter
    from meter.mch_meter import MchMeter
    from meter.mch_grad_meter import MchGradMeter
    from meter.punet_meter import PUNetMeter
    from meter.sqd_lstm_meter import SqdLstmMeter
    from meter.restormer_meter import RestormerMeter
    from meter.uformer_meter import UformerMeter
    from meter.wav_meter import WavMeter
    from meter.dlpu_meter import DLPUMeter
    from meter.u3net_meter import U3NetMeter
    # Import MModel
    from model.sqd_lstm_mmodel import SqdLstmMModel
    from model.restormer_mmodel import RestormerMModel
    from model.uformer_mmodel import UformerMModel
    from model.dlpu_mmodel import DLPUMModel
    from model.punet_mmodel import PUNetMModel
    from model.u3net_mmodel import U3NetMModel
    main()