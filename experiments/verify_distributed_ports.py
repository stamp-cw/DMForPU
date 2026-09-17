"""Two CPU/Gloo ranks exercise U3Net phase/checkpoint plumbing with a tiny network."""
import json
import logging
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.multiprocessing as mp

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TinyUnroller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, gradients, cond, x, a):
        prediction = self.conv(gradients[..., 0] + gradients[..., 1])
        return prediction, [prediction]


def worker(rank, port, output):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2',
                      MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), USE_LIBUV='0', OMP_NUM_THREADS='1',
                      CUDA_VISIBLE_DEVICES='')
    torch.set_num_threads(1)
    from accelerate import Accelerator
    from model.u3net_mmodel import U3NetMModel
    from model.optimizer import OptimizerFN
    from run.train_multi_model import ModelTrainer, EpochFN
    from meter.punet_meter import PUNetMeter
    from selector.optimizer_selector import _OPTIMIZERS
    accelerator = Accelerator(cpu=True)
    # Accelerate 1.15 passes CUDA device_ids even for a CPU/Gloo barrier.
    # Keep a real collective here, without the inapplicable GPU indices.
    accelerator.wait_for_everyone = lambda: torch.distributed.barrier()
    cfg = NS(
        training=NS(device='cpu', amp=False, brand_new_epochs=502, distill_epochs=2,
                    distill_start_epoch=500, use_all_data=False),
        optim=NS(optimizer='Adam', lr=.001, beta1=.9, eps=1e-8, weight_decay=0,
                 scheduler='exponential', gamma=.99, warmup=0, grad_clip=1),
        data=NS(noise_snr=30), loss_type=NS(name='U3NetLoss'), writer=None,
        mode='train_multi_model', accelerator=accelerator,
        io=NS(use_tensorboard=False, use_wandb=False, out_ckpt_path=str(output), out_ckpt_filename_prefix='ddp'),
    )
    wrapper = U3NetMModel.__new__(U3NetMModel)
    wrapper.config=cfg; wrapper.device='cpu'; wrapper.model=TinyUnroller()
    wrapper.teacher_model=None; wrapper.is_distilling=False
    meter=PUNetMeter(cfg); main_meter=PUNetMeter(cfg)
    meter.is_record=False; main_meter.is_record=False; cfg.train_meter=meter
    trainer=ModelTrainer.__new__(ModelTrainer)
    trainer.config=cfg; trainer.accelerator=accelerator; trainer.mmodel=wrapper
    trainer.optimizer=_OPTIMIZERS(cfg)(wrapper.optimize_parameters)
    trainer.logger=logging.getLogger('ddp-verification')
    trainer.meter=meter; trainer.main_meter=main_meter
    trainer.epoch_fn=EpochFN(OptimizerFN(cfg),cfg)
    trainer.start_epoch=499;trainer.end_epoch=502;trainer.acc_batch=0
    samples=[{'wrapped':torch.randn(1,8,8), 'unwrapped':torch.randn(1,8,8)} for _ in range(4)]
    trainer.data_loader=NS(train_loader=torch.utils.data.DataLoader(samples,batch_size=1))
    trainer._record_and_evaluate=lambda: trainer._save_state(trainer.epoch)
    trainer._train()
    assert wrapper.is_distilling
    assert not isinstance(wrapper.teacher_model,torch.nn.parallel.DistributedDataParallel)
    assert all(not p.requires_grad for p in wrapper.teacher_model.parameters())
    teacher_vector=torch.cat([p.detach().flatten() for p in wrapper.teacher_model.parameters()])
    gathered=accelerator.gather(teacher_vector).reshape(2,-1)
    torch.testing.assert_close(gathered[0],gathered[1])
    # A fresh trainer restores both optimizer and frozen teacher from the DDP checkpoint.
    cfg.io.latest_checkpoint_file_path=str(output/'ddp_501.pth')
    wrapper.model=TinyUnroller();wrapper.teacher_model=None;wrapper.is_distilling=False
    trainer.device='cpu';trainer.optimizer=_OPTIMIZERS(cfg)(wrapper.optimize_parameters)
    trainer._load_state()
    assert wrapper.is_distilling and not wrapper.configure_training_phase(501)
    restored=torch.cat([p.detach().flatten() for p in wrapper.teacher_model.parameters()])
    torch.testing.assert_close(restored,teacher_vector)
    if rank==0:
        (output/'ddp_cpu.json').write_text(json.dumps({'ranks':2,'backend':'gloo',
            'epochs':[499,500,501],'teacher_consistent_across_ranks':True,
            'teacher_restored':True,'lr':trainer.optimizer.param_groups[0]['lr']},indent=2))
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    output=ROOT/'experiments/results/port_fixes'
    output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    mp.spawn(worker,args=(port,output),nprocs=2,join=True)
