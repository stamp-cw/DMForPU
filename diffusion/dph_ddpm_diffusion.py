import torch
from diffusers import UNet2DConditionModel, DDPMScheduler, ControlNetModel

from model.model_setup import ModelSetup
from model.sec_model_setup import SecModelSetup
from selector.diffusion_selector import register_diffusion
import tqdm

from utils.util import ExtraTokenCondition, UNetFeatureHook


@register_diffusion(name='DphDDPMDiffusion')
class DphDDPMDiffusion:
    def __init__(self, config):
        self.config = config
        self.device = config.training.device

        self.unet = ModelSetup(self.config, config.logger).model.to(self.device)

        self.model = self.unet

        self.aux_unet = SecModelSetup(self.config, config.logger).model

        self.load_checkpoint()

        # load aux_unet weights to aux_unet
        # self.aux_unet.load_state_dict(torch.load(self.config.model.aux_unet_path, map_location=self.device))
        self.aux_unet.eval()

        self.scheduler = DDPMScheduler(num_train_timesteps=config.diffusion.num_train_timesteps)


    def unwrap_model(self, model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model


    def load_checkpoint(self):
        # self.logger.info(f"Loading checkpoint from {self.config.io.sampling_ckpt_file_path}")
        # loaded_state = torch.load(self.config.io.sampling_ckpt_file_path, map_location=self.device, weights_only=True)
        # self.mmodel.model.load_state_dict(loaded_state['model'], strict=True)
        # load_weights = r"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Big/model_weights/weights.pth"
        load_weights = r"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/assets/SyntheticPUMat128Test/AuxUNet/ckpt/epoch_100.pth"
        checkpoint = torch.load(load_weights, map_location='cpu')
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        self.aux_unet.load_state_dict(new_state_dict['state_dict'])
        # self.aux_unet.load_state_dict(checkpoint['state_dict'])

    def setup_train(self):
        self.model.train()

    def setup_eval(self):
        self.model.eval()

    def setup_data(self, batch_dict):
        self.wrapped = batch_dict["wrapped"].to(self.device)
        # self.wrapped_cond = batch_dict["wrapped_cond"].to(self.device)
        self.gt_unwrapped = batch_dict["unwrapped"].to(self.device)
        self.gt_unwrapped_norm = batch_dict["unwrapped_neg_norm"].to(self.device)

    def train_sample(self, t):

        # unet = UNet(config).to(device)

        aux_net = self.unwrap_model(self.aux_unet)

        _, d_feats, u_feats = aux_net(self.wrapped)

        self.noise = torch.randn_like(self.gt_unwrapped_norm).to(self.device)
        self.noisy = self.scheduler.add_noise(self.gt_unwrapped_norm, self.noise, t).to(self.device)
        self.noise_pred = self.model(self.noisy, t, feats=True, down_feats=d_feats, up_feats=u_feats)
        self.pred_unwrapped_neg_norm = self.scheduler.step(self.noise_pred, t[0].cpu(), self.noisy).pred_original_sample
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.k_max - self.config.data.k_min)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    def infer_sample(self):

        scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_infer_timesteps)

        aux_net = self.unwrap_model(self.aux_unet)

        _, d_feats, u_feats = aux_net(self.wrapped)

        x = torch.randn_like(self.wrapped).to(self.device)
        for t in tqdm.tqdm(scheduler.timesteps, desc="Sampling"):
            self.noise_pred = self.model(self.noisy, t, feats=True, down_feats=d_feats, up_feats=u_feats)
            x = scheduler.step(self.noise_pred, t, x).prev_sample
        self.pred_unwrapped_neg_norm = x
        self.pred_unwrapped_norm = (self.pred_unwrapped_neg_norm + 1) / 2
        self.pred_unwrapped = self.pred_unwrapped_norm * (2 * torch.pi * self.config.data.k_max - self.config.data.k_min)
        self.diff_unwrapped = self.gt_unwrapped - self.pred_unwrapped

    @property
    def optimize_parameters(self):
        return self.model.parameters()
