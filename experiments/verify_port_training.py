"""Short training verification; Accelerate mode saves a verification checkpoint."""
import argparse
import gc
import importlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from utils.util import dict2namespace
from model.mmodel_setup import MModelSetup
from meter.meter_setup import MeterSetup
from model.optimizer import OptimizerFN
from selector.optimizer_selector import _OPTIMIZERS
from run.train_model import EpochFN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accelerate', action='store_true', help='Exercise Accelerate U3Net phase transition and checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    torch.set_num_threads(4)
    for module in ('model.unet.dlpu', 'model.lstm.sqd_lstm', 'model.transformer.uformer',
                   'model.transformer.restormer', 'model.u3net.u3net'):
        importlib.import_module(module)
    for method in ('dlpu', 'sqd_lstm', 'uformer', 'restormer', 'u3net'):
        importlib.import_module(f'model.{method}_mmodel')
        importlib.import_module(f'meter.{method}_meter')
    importlib.import_module('meter.punet_meter')
    path = sorted((ROOT/'data/SyntheticPUMat128Big/train_in').glob('*.mat'))[0]
    batch = {
        'wrapped': torch.from_numpy(scipy.io.loadmat(path)['input']).float()[None, None],
        'unwrapped': torch.from_numpy(scipy.io.loadmat(path.parent.parent/'train_gt'/path.name)['gt']).float()[None, None],
    }
    results = {'sample': str(path.relative_to(ROOT)), 'shape': list(batch['wrapped'].shape), 'methods': {}}
    methods = ('u3net',) if args.accelerate else ('dlpu', 'sqd_lstm', 'uformer', 'restormer', 'u3net')
    for method in methods:
        cfg = dict2namespace(yaml.safe_load((ROOT/f'configs/{method}_synpu_128_big.yaml').read_text(encoding='utf-8')))
        cfg.mode = 'train_model'; cfg.writer = None; cfg.logger = logging.getLogger('port-smoke')
        cfg.io = SimpleNamespace(use_tensorboard=False, use_wandb=False)
        # Skip warmup only in this short verification so an update is observable.
        cfg.optim.warmup = 0
        wrapper = MModelSetup(cfg, cfg.logger).mmodel
        meter = MeterSetup(cfg, cfg.logger).meter
        cfg.train_meter = meter
        optimizer = _OPTIMIZERS(cfg)(wrapper.optimize_parameters)
        if args.accelerate:
            from accelerate import Accelerator
            from run.train_multi_model import EpochFN as MultiEpoch, ModelTrainer
            accelerator = Accelerator(mixed_precision='fp16')
            cfg.accelerator = accelerator
            wrapper.model, optimizer = accelerator.prepare(wrapper.model, optimizer)
            epoch_fn = MultiEpoch(OptimizerFN(cfg), cfg)
            trainer = ModelTrainer.__new__(ModelTrainer)
            trainer.config = cfg; trainer.mmodel = wrapper; trainer.accelerator = accelerator
            trainer.logger = cfg.logger; trainer.optimizer = optimizer
            main_meter = MeterSetup(cfg, cfg.logger).meter
        else:
            epoch_fn = EpochFN(OptimizerFN(cfg), cfg)
        phases = (499, 500) if method == 'u3net' else (1,)
        outcomes = []
        for epoch in phases:
            if args.accelerate:
                trainer._configure_training_phase(epoch)
                optimizer = trainer.optimizer
            elif hasattr(wrapper, 'configure_training_phase') and wrapper.configure_training_phase(epoch):
                optimizer = _OPTIMIZERS(cfg)(wrapper.optimize_parameters)
            wrapper.setup_train()
            parameter = next(p for p in wrapper.model.parameters() if p.requires_grad)
            before = parameter.detach().clone()
            for attempt in range(16):
                if args.accelerate:
                    epoch_fn(accelerator, wrapper, optimizer, meter, main_meter, epoch, dict(batch))
                else:
                    epoch_fn(wrapper, optimizer, meter, epoch, dict(batch))
                if not torch.isfinite(torch.as_tensor(meter.batch_metric_dict['loss'])).all():
                    raise AssertionError(f'{method}: non-finite loss')
                if not torch.equal(parameter.detach(), before):
                    break
            else:
                raise AssertionError(f'{method}: no update after scaler warmup')
            assert all(torch.isfinite(p).all() for p in wrapper.model.parameters())
            outcomes.append({'epoch': epoch, 'loss': float(meter.batch_metric_dict['loss']),
                             'attempts': attempt+1, 'lr': float(optimizer.param_groups[0]['lr']),
                             'distillation': getattr(wrapper, 'is_distilling', False)})
        results['methods'][method] = outcomes
        print(method, outcomes, flush=True)
        if args.accelerate:
            cfg.io.out_ckpt_path = str(args.output.parent)
            cfg.io.out_ckpt_filename_prefix = 'verification_u3net'
            args.output.parent.mkdir(parents=True, exist_ok=True)
            trainer._save_state(500)
            checkpoint = args.output.parent/'verification_u3net_500.pth'
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert 'teacher_model' in state and 'scaler' in state
            assert not any(k.startswith('module.') for k in state['model'])
            results['checkpoint'] = str(checkpoint)
        del wrapper, optimizer, epoch_fn, meter, parameter, before
        gc.collect(); torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
