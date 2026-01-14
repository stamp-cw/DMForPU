'''
Mainly based on https://github.com/yang-song/score_sde_pytorch/blob/main/datasets.py
'''

import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from functools import cached_property

# from datasets.CustomEval import CustomEval
from datasets.CustomTrain import CustomTrain
from datasets.FRBS import FRBS
# from datasets.HRSID import HRSID
# from datasets.SARBuD import SARBuD

from datasets.InSARDLPUMat import InSARDLPUMat

from torchvision.transforms import v2 as T


class BaseDataLoader:
    def __init__(self, config):
        self.config = config

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

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        # transform_list.append(transforms.Resize(
        #     (self.config.data.image_size, self.config.data.image_size),
        #     interpolation=transforms.InterpolationMode.BICUBIC))
        # if self.config.data.random_flip:
        #     transform_list.append(transforms.RandomHorizontalFlip())

        # 随机亮度、对比度、饱和度、色调
        # transform_list.append(transforms.ColorJitter(
        #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        # # 仅随机对比度
        # transform_list.append(transforms.ColorJitter(contrast=0.2))
        # # 高斯模糊
        # transform_list.append(transforms.GaussianBlur(
        #     kernel_size=3, sigma=(0.1, 2.0)))
        # # 随机噪声（可自定义）
        # transform_list.append(transforms.AdditiveGaussianNoise(mean=0.0, std=0.05, p=0.5))

        # 转为张量0-1
        transform_list.append(transforms.ToTensor())

        # # Normalize
        # transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        if uniform_dequantization:
            transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        # # 随机伽马变换
        # transform_list.append(transforms.RandomAdjustSharpness(
        #     sharpness_factor=2, p=0.5))

        # # 随机锐化
        # transform_list.append(transforms.RandomAdjustSharpness(
        #     sharpness_factor=2, p=0.5))
        # # 随机自动对比度
        # transform_list.append(transforms.RandomAutocontrast(p=0.5))
        # # 随机直方图均衡化
        # transform_list.append(transforms.RandomEqualize(p=0.5))
        #
        # # 随机反转高亮像素
        # # transform_list.append(transforms.RandomErasing())
        # transform_list.append(transforms.RandomSolarize(threshold=192.0, p=0.5))
        # # 随机遮挡
        # transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))

        return transforms.Compose(transform_list)

    @property
    def gt_transform(self, uniform_dequantization=False):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        # transform_list.append(transforms.Resize(
        #     (self.config.data.image_size, self.config.data.image_size),
        #     interpolation=transforms.InterpolationMode.BICUBIC))
        # if self.config.data.random_flip:
        #     transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        # transform_list.append(transforms.PILToTensor())

        if uniform_dequantization:
            transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        return transforms.Compose(transform_list)

    @property
    def joint_transform(self):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        # 先调整到目标大小
        transform_list.append(T.Resize(
            (self.config.data.image_size, self.config.data.image_size),
            interpolation=T.InterpolationMode.BICUBIC))
        # 随机水平翻转
        transform_list.append(T.RandomHorizontalFlip())
        # 随机垂直翻转
        transform_list.append(T.RandomVerticalFlip())
        # 随机裁剪 + 缩放回目标大小
        transform_list.append(T.RandomResizedCrop(size=(self.config.data.image_size, self.config.data.image_size)))
        # 随机旋转
        transform_list.append(T.RandomRotation(degrees=15))
        # 综合仿射变换（旋转 + 平移 + 缩放 + 错切）
        transform_list.append(T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10))
        # # 随机透视变换（更强的几何扰动）
        # transform_list.append(T.RandomPerspective(distortion_scale=0.1, p=0.5))
        # # 转为张量0-1
        # transform_list.append(T.ToTensor())
        #
        # if uniform_dequantization:
        #     transform_list.append(T.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        return T.Compose(transform_list)

    @property
    def eval_transform(self, uniform_dequantization=False):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        # transform_list.append(transforms.ColorJitter(contrast=0.2))
        # 转为张量0-1
        transform_list.append(transforms.ToTensor())
        # # Normalize
        # transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        if uniform_dequantization:
            transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        # # 随机伽马变换
        # transform_list.append(transforms.RandomAdjustSharpness(
        #     sharpness_factor=2, p=0.5))

        # # 随机锐化
        # transform_list.append(transforms.RandomAdjustSharpness(
        #     sharpness_factor=2, p=0.5))
        # # 随机自动对比度
        # transform_list.append(transforms.RandomAutocontrast(p=0.5))
        # # 随机直方图均衡化
        # transform_list.append(transforms.RandomEqualize(p=0.5))
        #
        # # 随机反转高亮像素
        # # transform_list.append(transforms.RandomErasing())
        # transform_list.append(transforms.RandomSolarize(threshold=192.0, p=0.5))
        # # 随机遮挡
        # transform_list.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))

        return transforms.Compose(transform_list)

    @property
    def eval_gt_transform(self, uniform_dequantization=False):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        transform_list.append(transforms.ToTensor())

        if uniform_dequantization:
            transform_list.append(transforms.Lambda(lambda x: (x * 255 + torch.rand_like(x)) / 256))

        return transforms.Compose(transform_list)

    @property
    def eval_joint_transform(self):

        self.batch_size = self.config.training.batch_size
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.batch_size % num_devices != 0:
            raise ValueError(f'Batch size {self.batch_size} must be divisible by the number of devices {num_devices}')

        transform_list = []
        # 先调整到目标大小
        transform_list.append(T.Resize(
            (self.config.data.image_size, self.config.data.image_size),
            interpolation=T.InterpolationMode.BICUBIC))

        return T.Compose(transform_list)


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
        local_name = name if name is not None else cls.__name__
        if local_name in _DATA_LOADERS:
            raise ValueError(f'Already registered data loader with name: {local_name}')
        _DATA_LOADERS[local_name] = cls
        return cls

    return _register(cls) if cls is not None else _register


@register_data_loader(name='SARBuD')
class SARBuDDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def train_dataset(self):
        return SARBuD(root=self.config.io.in_dataset_path, train=True,
                      transform=self.transform,
                      target_transform=self.gt_transform,
                      joint_transform=self.joint_transform)

    @cached_property
    def eval_dataset(self):
        return SARBuD(root=self.config.io.in_dataset_path, train=False,
                      transform=self.eval_transform,
                      target_transform=self.eval_gt_transform,
                      joint_transform=self.eval_joint_transform)

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)


@register_data_loader(name='HRSID')
class HRSIDDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def train_dataset(self):
        return HRSID(root=self.config.io.in_dataset_path, split='train',
                     transform=self.transform,
                     target_transform=self.gt_transform,
                     joint_transform=self.joint_transform)

    @cached_property
    def eval_dataset(self):
        return HRSID(root=self.config.io.in_dataset_path, split='train_test',
                     transform=self.eval_transform,
                     target_transform=self.eval_gt_transform,
                     joint_transform=self.eval_joint_transform)

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)


@register_data_loader(name='FRBS')
class FRBSDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)
        self.fold = config.data.fold

    @property
    def train_dataset(self):
        return FRBS(root=self.config.io.in_dataset_path, split='train',
                    transform=self.transform,
                    target_transform=self.gt_transform,
                    joint_transform=self.joint_transform,
                    fold=self.fold, num_folds=self.config.data.num_folds)

    @property
    def eval_dataset(self):
        return FRBS(root=self.config.io.in_dataset_path, split='val',
                    transform=self.eval_transform,
                    target_transform=self.eval_gt_transform,
                    joint_transform=self.eval_joint_transform,
                    fold=self.fold, num_folds=self.config.data.num_folds)

    @cached_property
    def test_dataset(self):
        return FRBS(root=self.config.io.in_dataset_path, split='test',
                    transform=self.eval_transform,
                    target_transform=self.eval_gt_transform,
                    joint_transform=self.eval_joint_transform)

    @property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])

    @cached_property
    def all_fold_train_dataset_list(self):
        all_fold_dataset = []
        for fold in range(self.config.data.num_folds):
            self.fold = fold
            fold_dataset = FRBS(root=self.config.io.in_dataset_path, split='train',
                                transform=self.transform,
                                target_transform=self.gt_transform,
                                joint_transform=self.joint_transform,
                                fold=self.fold, num_folds=self.config.data.num_folds)
            all_fold_dataset.append(fold_dataset)
        return all_fold_dataset

    @cached_property
    def all_fold_train_loader_list(self):
        all_fold_loader = []
        for fold in range(self.config.data.num_folds):
            fold_loader = DataLoader(self.all_fold_train_dataset_list[fold], batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)
            all_fold_loader.append(fold_loader)
        return all_fold_loader

    @cached_property
    def all_fold_eval_dataset_list(self):
        all_fold_dataset = []
        for fold in range(self.config.data.num_folds):
            self.fold = fold
            fold_dataset = FRBS(root=self.config.io.in_dataset_path, split='val',
                                transform=self.transform,
                                target_transform=self.gt_transform,
                                joint_transform=self.joint_transform,
                                fold=self.fold, num_folds=self.config.data.num_folds)
            all_fold_dataset.append(fold_dataset)
        return all_fold_dataset

    @cached_property
    def all_fold_eval_loader_list(self):
        all_fold_loader = []
        for fold in range(self.config.data.num_folds):
            fold_loader = DataLoader(self.all_fold_eval_dataset_list[fold], batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)
            all_fold_loader.append(fold_loader)
        return all_fold_loader

    @property
    def train_loader(self):
        self.fold = self.config.data.fold
        tmp_loader = self.all_fold_train_loader_list[self.fold]
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
        #                   num_workers=self.config.data.num_workers, pin_memory=True, drop_last=True)
        return tmp_loader

    @property
    def eval_loader(self):
        self.fold = self.config.data.fold
        tmp_loader = self.all_fold_eval_loader_list[self.fold]
        # return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False,
        #                   num_workers=self.config.data.num_workers, pin_memory=True)
        return tmp_loader

    @cached_property
    def test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)


@register_data_loader(name='CustomTrain')
class CustomTrainDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def train_dataset(self):
        return CustomTrain(root=self.config.io.in_dataset_path, train=True,
                           transform=self.transform,
                           target_transform=self.gt_transform,
                           joint_transform=self.joint_transform)

    @cached_property
    def eval_dataset(self):
        return CustomTrain(root=self.config.io.in_dataset_path, train=False,
                           transform=self.eval_transform,
                           target_transform=self.eval_gt_transform,
                           joint_transform=self.eval_joint_transform)

    @cached_property
    def all_dataset(self):
        return torch.utils.data.ConcatDataset([self.train_dataset, self.eval_dataset])

    @cached_property
    def train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)


@register_data_loader(name='CustomEval')
class CustomEvalDataLoader(BaseDataLoader):
    def __init__(self, config):
        super().__init__(config)

    @cached_property
    def eval_dataset(self):
        return CustomEval(root=self.config.io.in_dataset_path,
                          transform=self.eval_transform)

    @cached_property
    def all_dataset(self):
        return self.eval_dataset

    @cached_property
    def eval_loader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)

    @cached_property
    def all_loader(self):
        return DataLoader(self.all_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.config.data.num_workers, pin_memory=True)