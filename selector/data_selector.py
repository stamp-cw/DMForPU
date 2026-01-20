'''
Mainly based on https://github.com/yang-song/score_sde_pytorch/blob/main/datasets.py
'''

import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import cached_property

from dataset.InSARDLPUMat import InSARDLPUMat

from torchvision.transforms import v2 as T

from dataset.InSARDLPUMatCut import InSARDLPUMatCut
from dataset.SyntheticData import SyntheticData
from dataset.SyntheticPUMat import SyntheticPUMat
from dataset.SyntheticPUMatCut import SyntheticPUMatCut


class BaseDataLoader:
    def __init__(self, config):
        self.config = config
        if config.mode == 'sample':
            self.batch_size = self.config.sampling.batch_size
        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

    def data_scaler(self, x):
        if self.config.data.centered:
            return x * 2 - 1
        else:
            return x

    def data_inverse_scaler(self, x):
        if self.config.data.centered:
            return (x + 1) / 2
        else:
            return x

    @property
    def transform(self, uniform_dequantization=False):

        # transform_list = []
        # # transform_list.append(transforms.Resize(
        # #     (self.config.data.image_size, self.config.data.image_size),
        # #     interpolation=transforms.InterpolationMode.BICUBIC))
        # # if self.config.data.random_flip:
        # #     transform_list.append(transforms.RandomHorizontalFlip())
        #
        # # 随机亮度、对比度、饱和度、色调
        # # transform_list.append(transforms.ColorJitter(
        # #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # # # 仅随机对比度
        # # transform_list.append(transforms.ColorJitter(contrast=0.2))
        # # # 高斯模糊
        # # transform_list.append(transforms.GaussianBlur(
        # #     kernel_size=3, sigma=(0.1, 2.0)))
        # # # 随机噪声（可自定义）
        # # transform_list.append(transforms.AdditiveGaussianNoise(mean=0.0, std=0.05, p=0.5))
        #
        # # 转为张量0-1
        # transform_list.append(transforms.ToTensor())
        #
        # # # Normalize
        # # transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        #
        # if uniform_dequantization:
        #     transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))
        #
        # # # 随机伽马变换
        # # transform_list.append(transforms.RandomAdjustSharpness(
        # #     sharpness_factor=2, p=0.5))
        #
        # # # 随机锐化
        # # transform_list.append(transforms.RandomAdjustSharpness(
        # #     sharpness_factor=2, p=0.5))
        # # # 随机自动对比度
        # # transform_list.append(transforms.RandomAutocontrast(p=0.5))
        # # # 随机直方图均衡化
        # # transform_list.append(transforms.RandomEqualize(p=0.5))
        # #
        # # # 随机反转高亮像素
        # # # transform_list.append(transforms.RandomErasing())
        # # transform_list.append(transforms.RandomSolarize(threshold=192.0, p=0.5))
        # # # 随机遮挡
        # # transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))

        # return transforms.Compose(transform_list)
        return None

    @property
    def gt_transform(self, uniform_dequantization=False):

        # transform_list = []
        # # transform_list.append(transforms.Resize(
        # #     (self.config.data.image_size, self.config.data.image_size),
        # #     interpolation=transforms.InterpolationMode.BICUBIC))
        # # if self.config.data.random_flip:
        # #     transform_list.append(transforms.RandomHorizontalFlip())
        # transform_list.append(transforms.ToTensor())
        # # transform_list.append(transforms.PILToTensor())
        #
        # if uniform_dequantization:
        #     transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        # return transforms.Compose(transform_list)
        return None

    @property
    def joint_transform(self):

        # transform_list = []
        # # 先调整到目标大小
        # transform_list.append(T.Resize(
        #     (self.config.data.image_size, self.config.data.image_size),
        #     interpolation=T.InterpolationMode.BICUBIC))
        # # 随机水平翻转
        # transform_list.append(T.RandomHorizontalFlip())
        # # 随机垂直翻转
        # transform_list.append(T.RandomVerticalFlip())
        # # 随机裁剪 + 缩放回目标大小
        # transform_list.append(T.RandomResizedCrop(size=(self.config.data.image_size, self.config.data.image_size)))
        # # 随机旋转
        # transform_list.append(T.RandomRotation(degrees=15))
        # # 综合仿射变换（旋转 + 平移 + 缩放 + 错切）
        # transform_list.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
        # # # 随机透视变换（更强的几何扰动）
        # # transform_list.append(T.RandomPerspective(distortion_scale=0.1, p=0.5))
        # # # 转为张量0-1
        # # transform_list.append(T.ToTensor())
        # #
        # # if uniform_dequantization:
        # #     transform_list.append(T.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        # return T.Compose(transform_list)
        return None

    @property
    def eval_transform(self, uniform_dequantization=False):
        # transform_list = []
        # # transform_list.append(transforms.ColorJitter(contrast=0.2))
        # # 转为张量0-1
        # transform_list.append(transforms.ToTensor())
        # # # Normalize
        # # transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        #
        # if uniform_dequantization:
        #     transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))
        #
        # # # 随机伽马变换
        # # transform_list.append(transforms.RandomAdjustSharpness(
        # #     sharpness_factor=2, p=0.5))
        #
        # # # 随机锐化
        # # transform_list.append(transforms.RandomAdjustSharpness(
        # #     sharpness_factor=2, p=0.5))
        # # # 随机自动对比度
        # # transform_list.append(transforms.RandomAutocontrast(p=0.5))
        # # # 随机直方图均衡化
        # # transform_list.append(transforms.RandomEqualize(p=0.5))
        # #
        # # # 随机反转高亮像素
        # # # transform_list.append(transforms.RandomErasing())
        # # transform_list.append(transforms.RandomSolarize(threshold=192.0, p=0.5))
        # # # 随机遮挡
        # # transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))

        # return transforms.Compose(transform_list)
        return None

    @property
    def eval_gt_transform(self, uniform_dequantization=False):

        # transform_list = []
        # transform_list.append(transforms.ToTensor())
        #
        # if uniform_dequantization:
        #     transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        # return transforms.Compose(transform_list)
        return None

    @property
    def eval_joint_transform(self):

        # transform_list = []
        # # 先调整到目标大小
        # transform_list.append(T.Resize(
        #     (self.config.data.image_size, self.config.data.image_size),
        #     interpolation=T.InterpolationMode.BICUBIC))

        # return T.Compose(transform_list)
        return None


class DataLoaderRegistry(dict):
    def __getitem__(self, key):
        if not isinstance(key, str):
            name = key.data.name
            return super().__getitem__(name)(key)
        return super().__getitem__(key)

    def __call__(self, config):
        return self[config]


_DATA_LOADERS = DataLoaderRegistry()


def register_data_loader(cls=None, *, name=None):
    def _register(cls):
        if isinstance(name, list):
            for local_name in name:
                if local_name in _DATA_LOADERS:
                    raise ValueError(f'Already registered data loader with name: {local_name}')
                _DATA_LOADERS[local_name] = cls
        else:
            local_name = name if name is not None else cls.__name__
            if local_name in _DATA_LOADERS:
                raise ValueError(f'Already registered data loader with name: {local_name}')
            _DATA_LOADERS[local_name] = cls
        return cls
    return _register(cls) if cls is not None else _register


@register_data_loader(name=['InSARDLPUMat','InSARDLPUMat256Big','InSARDLPUMat256Small','InSARDLPUMat256Test'])
class InSARDLPUMatDataLoader(BaseDataLoader):

    @cached_property
    def train_dataset(self):
        return InSARDLPUMat(root=self.config.io.in_dataset_path, split='train',
                    transform=self.transform,
                    target_transform=self.gt_transform,
                    joint_transform=self.joint_transform,
                    scale_k=self.config.data.scale_k,
                            )

    @cached_property
    def test_dataset(self):
        return InSARDLPUMat(root=self.config.io.in_dataset_path, split='test',
                    transform=self.eval_transform,
                    target_transform=self.eval_gt_transform,
                    joint_transform=self.eval_joint_transform,
                    scale_k=self.config.data.scale_k,
                            )

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)


@register_data_loader(name=['InSARDLPUMatCut','InSARDLPUMatCut32','InSARDLPUMatCut32Test','InSARDLPUMatCut64','InSARDLPUMatCut64Test'])
class InSARDLPUMatCutDataLoader(BaseDataLoader):

    @cached_property
    def train_dataset(self):
        return InSARDLPUMatCut(root=self.config.io.in_dataset_path, split='train',
                            transform=self.transform,
                            target_transform=self.gt_transform,
                            joint_transform=self.joint_transform,
                            scale_k=self.config.data.scale_k,
                            )

    @cached_property
    def test_dataset(self):
        return InSARDLPUMatCut(root=self.config.io.in_dataset_path, split='test',
                            transform=self.eval_transform,
                            target_transform=self.eval_gt_transform,
                            joint_transform=self.eval_joint_transform,
                            scale_k=self.config.data.scale_k,
                            )

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

@register_data_loader(name=['SyntheticPUMat','SyntheticPUMat64Test','SyntheticPUMat128Big','SyntheticPUMat128Small','SyntheticPUMat128Test'])
class SyntheticPUMatDataLoader(BaseDataLoader):

    @cached_property
    def train_dataset(self):
        return SyntheticPUMat(root=self.config.io.in_dataset_path, split='train',
                            transform=self.transform,
                            target_transform=self.gt_transform,
                            joint_transform=self.joint_transform,
                              k_max=self.config.data.k_max,
                              k_min=self.config.data.k_min
                            )

    @cached_property
    def test_dataset(self):
        return SyntheticPUMat(root=self.config.io.in_dataset_path, split='test',
                            transform=self.eval_transform,
                            target_transform=self.eval_gt_transform,
                            joint_transform=self.eval_joint_transform,
                              k_max=self.config.data.k_max,
                              k_min=self.config.data.k_min
                            )

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

@register_data_loader(name=['SyntheticPUMatCut','SyntheticPUMatCut128Big','SyntheticPUMatCut32','SyntheticPUMatCut32Big','SyntheticPUMatCut32Test','SyntheticPUMatCut64','SyntheticPUMatCut64Test'])
class SyntheticPUMatCutMatDataLoader(BaseDataLoader):

    @cached_property
    def train_dataset(self):
        return SyntheticPUMatCut(root=self.config.io.in_dataset_path, split='train',
                              transform=self.transform,
                              target_transform=self.gt_transform,
                              joint_transform=self.joint_transform,
                                 k_max=self.config.data.k_max,
                                 k_min=self.config.data.k_min
                              )

    @cached_property
    def test_dataset(self):
        return SyntheticPUMatCut(root=self.config.io.in_dataset_path, split='test',
                              transform=self.eval_transform,
                              target_transform=self.eval_gt_transform,
                              joint_transform=self.eval_joint_transform,
                                 k_max=self.config.data.k_max,
                                 k_min=self.config.data.k_min
                              )

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

@register_data_loader(name='SyntheticData')
class SyntheticDataDataLoader(BaseDataLoader):

    @cached_property
    def train_dataset(self):
        return SyntheticData(root=self.config.io.in_dataset_path, split='train',
                                 transform=self.transform,
                                 target_transform=self.gt_transform,
                                 joint_transform=self.joint_transform,
                                 scale_k=self.config.data.scale_k,
                                 )

    @cached_property
    def test_dataset(self):
        return SyntheticData(root=self.config.io.in_dataset_path, split='test',
                                 transform=self.eval_transform,
                                 target_transform=self.eval_gt_transform,
                                 joint_transform=self.eval_joint_transform,
                                 scale_k=self.config.data.scale_k,
                                 )

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)