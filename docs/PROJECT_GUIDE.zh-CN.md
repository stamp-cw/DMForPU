# DMForPU 新手上手指南

这份文档面向第一次接触本项目的人。目标不是一次讲完所有模型细节，而是让你先知道项目在解决什么问题、一次命令经过哪些代码、怎样准备数据和配置，以及遇到错误时先查哪里。

## 1. 项目是做什么的

DMForPU 是论文 **WWFDiff-PU: A Wavelet Frequency Decomposed Diffusion Model For Phase Unwrapping** 的实验代码。它研究的是相位解缠（phase unwrapping）：从被限制在一个周期内的缠绕相位，恢复连续的绝对相位。

仓库同时包含两类实验路线：

- **扩散模型路线**：以 `FduDDPMDiffusion` 为主，也保留 `WavDDPMDiffusion`、`MchDDPMDiffusion` 等实现。
- **普通监督模型路线**：包含 DLPU、PUNet、Restormer、Uformer、SQD-LSTM、U3Net 等基线。

你可以先把它理解成一个“配置驱动的实验框架”：大多数时候不需要改 `main.py`，而是选一份 `configs/*.yaml`，再指定训练、采样或验证模式。

## 2. 先认识目录

```text
DMForPU/
├─ main.py                 # 统一命令行入口
├─ configs/                # 实验配置，每份 YAML 描述一组实验
├─ dataset/                # Dataset：读取 MAT/TIF，生成训练所需张量
├─ selector/               # 注册表：把配置中的名字映射到 Python 类
├─ model/                  # 神经网络与普通模型包装器
├─ diffusion/              # 扩散训练、加噪和反向采样逻辑
├─ run/                    # train/sample/val/test 等流程控制器
├─ meter/                  # 指标计算与汇总
├─ utils/                  # 日志、指标和通用工具
├─ vae/                    # 可选的 VAE 实验代码
├─ data/                   # 本地数据，已被 .gitignore 忽略
├─ assets/                 # checkpoint、图像、指标和日志输出
├─ *.sh                    # 作者保存的实验命令集合
└─ requirements.txt        # 基础依赖列表
```

新手最值得先读的文件顺序是：

1. `configs/fdu_synpu_128_big.yaml`：先看一场实验由哪些参数组成。
2. `main.py`：看参数解析、配置加载和 mode 分发。
3. `run/train.py`：看训练循环。
4. `selector/data_selector.py`：看 `data.name` 如何选择数据加载器。
5. `diffusion/fdu_ddpm_diffusion.py`：最后再进入核心算法。

`article/`、`article_old/`、`article_old2/` 和 `diffusion/old/` 更像论文材料与历史实现，第一次阅读可以先跳过。

## 3. 一条命令是怎样运行的

以扩散模型训练为例：

```text
python main.py --config fdu_synpu_128_big.yaml --mode train
         │
         ├─ 读取 configs/fdu_synpu_128_big.yaml
         ├─ 计算 data/ 与 assets/ 下的输入输出路径
         ├─ 根据配置名从注册表选择组件
         │    ├─ data.name       -> DataLoader
         │    ├─ model.name      -> 网络
         │    ├─ diffusion.name  -> 扩散过程
         │    ├─ loss_type.name  -> 损失函数
         │    ├─ meter.name      -> 指标器
         │    └─ optim.optimizer -> 优化器
         └─ run/train.py -> 训练、记录指标、保存 checkpoint
```

这些注册表依赖装饰器，例如 `@register_diffusion(name='FduDDPMDiffusion')`。`main.py` 底部集中导入实现类，导入时装饰器执行，类才会进入注册表。因此，配置中的名字必须与注册名完全一致，大小写也不能错。

普通监督模型会走 `run/train_model.py`，并在网络外再套一层 `MModel` 包装器；扩散模型则由 `DiffusionSetup` 创建扩散对象，扩散对象内部再创建网络。

## 4. 准备运行环境

### 4.1 建议环境

仓库没有锁定 Python 和依赖版本。现有代码明显以 CUDA 训练为主，建议第一次使用：

- Python 3.10 或 3.11；
- NVIDIA GPU 与匹配的 CUDA 版 PyTorch；
- 在项目根目录执行所有命令。

创建独立环境：

```bash
conda create -n dmforpu python=3.10 -y
conda activate dmforpu
```

先根据显卡驱动，从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择正确的安装命令。不要直接照抄 `install.sh` 中的 CUDA 版本，除非你的机器确实与它匹配。

再安装仓库依赖：

```bash
pip install -r requirements.txt
pip install accelerate pyyaml tqdm pillow pywavelets einops bitsandbytes pytorch-msssim seaborn
```

第二条命令补充了源码或 `install.sh` 使用、但 `requirements.txt` 当前未完整列出的包。`bitsandbytes` 在 Windows 或部分显卡环境中可能安装失败；如果配置只使用普通 `Adam`，可以先不装，但当前 `model/optimizer.py` 在模块导入阶段仍会导入它，因此最省事的首次环境是 Linux/WSL2 或项目的 CUDA 容器环境。

### 4.2 验证安装

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available(), 'GPU数:', torch.cuda.device_count())"
python main.py --help
```

第一条命令应输出 PyTorch 版本，并显示 `CUDA: True`。第二条命令应打印 `--config`、`--mode` 等参数，而不是导入错误。

当前训练代码会调用 CUDA 专用接口，配置也默认写着 `device: cuda`，所以“能导入 PyTorch”不等于“能在纯 CPU 上训练”。

## 5. 准备数据

`main.py` 自动把输入根目录设成 `data/<data.name>/`。配置中的 `data.name` 同时也是数据加载器注册名，所以目录名和注册名通常保持一致。

### 5.1 SyntheticPUMat 系列

```text
data/SyntheticPUMat128Big/
├─ train_in/*.mat
├─ train_gt/*.mat
├─ test_in/*.mat
└─ test_gt/*.mat
```

- 输入与真值通过相同文件主名配对。
- 输入 MAT 中读取 `input` 字段，真值 MAT 中读取 `gt` 字段。
- 没有任何配对文件时会抛出 `FileNotFoundError`。

原始数据来源可参考 <https://github.com/kqwang/Phase_unwrapping_by_U-Net>。

### 5.2 InSARDLPUMat 系列

```text
data/InSARDLPUMat256Big/
├─ train_wrapped/*.mat
├─ train_absolute/*.mat
├─ test_wrapped/*.mat
└─ test_absolute/*.mat
```

- 输入 MAT 中读取 `input` 字段，真值 MAT 中读取 `output` 字段。
- 文件同样按主名配对。

原始数据来源可参考 <https://github.com/zhoulifan/InSAR-DLPU>。

### 5.3 R2AU TIF 系列

```text
data/R2AUTif256Big/
├─ noisy_phase/*.tif
└─ unwrapped_phase/*.tif
```

加载器从文件名末尾提取数字 ID 配对，再用固定随机种子按 9:1 划分训练集和测试集。

### 5.4 一个样本包含什么

常用 MAT/TIF 加载器最终返回类似下面的字典：

| 键 | 含义 | 常见形状 |
| --- | --- | --- |
| `wrapped` | 原始缠绕相位 | `[1, H, W]` |
| `unwrapped` | 绝对相位真值 | `[1, H, W]` |
| `wrapped_neg_norm` | 归一化到 `[-1, 1]` 的缠绕相位 | `[1, H, W]` |
| `unwrapped_neg_norm` | 归一化到 `[-1, 1]` 的真值 | `[1, H, W]` |
| `wrapped_cond` | `sin(wrapped)` 与 `cos(wrapped)` 拼接的条件 | `[2, H, W]` |

不同实验加载器可能再增加小波、噪声或梯度字段。读某个扩散实现前，先搜索它在 `setup_data()` 中取了哪些键。

## 6. 看懂配置文件

以 `configs/fdu_synpu_128_big.yaml` 为例，顶层分组的职责如下：

| 分组 | 作用 | 最常改的字段 |
| --- | --- | --- |
| `data` | 数据集、尺寸、归一化和小波参数 | `name`、`image_size`、`num_workers`、`k_min/k_max` |
| `model` | 扩散网络或基线网络 | `name`、通道数、block 配置 |
| `diffusion` | 扩散过程 | `name`、训练/推理步数、`prediction_type` |
| `loss_type` | 损失函数注册名 | `name` |
| `meter` | 训练与验证指标注册名 | `name` |
| `training` | 训练轮数、batch、保存频率 | `device`、`brand_new_epochs`、`batch_size`、`snapshot_freq` |
| `sampling` | 采样数量与随机种子 | `batch_size`、`total_samples`、`fix_seed` |
| `val` / `test` | 验证和测试 batch | `device`、`batch_size` |
| `optim` | 优化器与梯度设置 | `optimizer`、`lr`、`weight_decay`、`grad_clip` |
| `iio` | 实验记录开关 | `use_tensorboard`、`use_wandb`、`save_pth_to_wandb` |

文件名前缀也能帮助你快速筛选：`fdu_*` 是主要扩散实验，`dlpu_*`、`punet_*`、`restormer_*`、`uformer_*`、`sqd_lstm_*`、`u3net_*` 是普通模型基线。数据集通常写在中间，例如 `synpu_128_big`、`dlpu_256_big` 或 `r2au_256_big`。

第一次试跑可以直接使用 `configs/smoke_fdu_synpu_128.yaml`。它把训练轮数、batch、网络通道和推理步数都压小，并把 `data.num_workers` 设为 `0`，用于在单张 8 GB CUDA 显卡上检查流程。它不是论文实验配置，不能用来比较模型指标。

## 7. 跑通第一个实验

### 7.1 扩散模型训练

smoke 配置读取 `data/SyntheticPUMat128Test/`。请从 `SyntheticPUMat128Big` 中复制少量同名的输入/真值文件，保持第 5.1 节的四个子目录结构。当前验证使用了 2 对训练样本和 2 对测试样本。

然后运行：

```bash
python main.py --config smoke_fdu_synpu_128.yaml --mode train --training_from_scratch
```

`--config` 只接收 `configs/` 下的文件名，不能写成 `configs/smoke_fdu_synpu_128.yaml`。`--training_from_scratch` 表示忽略已有 checkpoint，从第 0 轮开始。

成功时，你会看到 epoch 进度和 loss 日志，并得到：

```text
assets/SyntheticPUMat128Test/FduDDPMDiffusion/
├─ ckpt/epoch_0.pth
└─ tb/
```

### 7.2 从断点继续训练

去掉 `--training_from_scratch`：

```bash
python main.py --config smoke_fdu_synpu_128.yaml --mode train
```

程序会扫描 `ckpt/epoch_*.pth`，取数字最大的 epoch，加载模型和优化器，并从下一轮继续。配置的 `training.continue_training_epochs` 表示本次再训练多少轮。

### 7.3 采样和验证

指定刚生成的 checkpoint：

```bash
python main.py --config smoke_fdu_synpu_128.yaml --mode sample --sampling_from_epoch 0
python main.py --config smoke_fdu_synpu_128.yaml --mode val --sampling_from_epoch 0
```

如果不传 `--sampling_from_epoch`，程序会自动选择最新 checkpoint。采样图像写到 `sample/<epoch>/`，验证结果写到 `val/<epoch>/`。

### 7.4 查看 TensorBoard

```bash
tensorboard --logdir assets/SyntheticPUMat128Test/FduDDPMDiffusion/tb --port 10086
```

浏览器打开 <http://localhost:10086>。如果配置中 `iio.use_tensorboard: false`，则不会生成这部分日志。

## 8. 命令行参考

### 8.1 参数

| 参数 | 必需 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--config FILE` | 是 | 无 | `configs/` 下的文件名 |
| `--mode MODE` | 是 | 无 | 选择运行流程，见下表 |
| `--user_logging_level` | 否 | `info` | `debug`、`info`、`warning` 或 `error` |
| `--training_from_scratch` | 否 | 关闭 | 从头训练，不加载自动找到的 checkpoint |
| `--sampling_from_epoch N` | 否 | 最新 | 采样、验证或测试时使用的 epoch |
| `--hyper` | 否 | 关闭 | 启用 W&B 超参数流程中的特殊配置处理 |
| `--debug` | 否 | 关闭 | 当前只被解析，主流程没有使用它改变行为 |

### 8.2 mode

| mode | 入口类 | 用途 |
| --- | --- | --- |
| `train` | `run.train.Trainer` | 单进程扩散模型训练 |
| `train_multi` | `run.train_multi.Trainer` | Accelerate 多卡扩散训练 |
| `sample` | `run.sample.Sampler` | 扩散模型采样 |
| `val` | `run.val.Valuator` | 扩散模型验证 |
| `test` | `run.test.Tester` | 扩散模型测试 |
| `train_model` | `run.train_model.ModelTrainer` | 普通监督模型训练 |
| `train_multi_model` | `run.train_multi_model.ModelTrainer` | 普通模型多卡训练 |
| `sample_model` | `run.sample_model.ModelSampler` | 普通模型采样 |
| `val_model` | `run.val_model.ModelValidator` | 普通模型验证 |
| `test_model` | `run.test_model.ModelTester` | 普通模型测试 |
| `train_vae` | `run.train_vae.VAETrainer` | VAE 训练 |
| `test_vae` | `run.test_vae.VAETester` | VAE 测试 |

## 9. 输出与 checkpoint

`main.py` 根据运行路线构造输出目录：

```text
assets/<data.name>/<diffusion.name>/   # 扩散路线
assets/<data.name>/<model.name>/       # 普通模型或 VAE 路线
```

常见内容：

| 路径 | 内容 |
| --- | --- |
| `ckpt/epoch_N.pth` | 模型和优化器状态 |
| `sample/N/` | 第 N 轮 checkpoint 的采样图与 `.pt` 数据 |
| `val/N/` | 验证指标和可选批次数据 |
| `tb/` | TensorBoard event 文件 |
| `wandb/` | W&B 本地运行目录 |

训练保存的 checkpoint 是一个字典，主要键为 `model` 和 `optimizer`。采样与验证通常只读取 `model`。代码能识别一部分带 `module.` 前缀的多卡权重，但不同流程的兼容处理并不完全一致。

## 10. 常见问题

### `ModuleNotFoundError`

先确认已激活正确环境，再对照第 4 节补装依赖。当前 `requirements.txt` 不是完整锁定文件，出现 `accelerate`、`yaml`、`tqdm`、`pywt`、`einops` 或 `bitsandbytes` 缺失并不意外。

### `No paired files found`

检查四件事：数据目录是否与 `data.name` 相同；子目录名是否正确；输入与真值文件主名是否相同；MAT 内部字段是否为加载器期望的名字。

### `Batch size ... must be divisible by the number of devices`

`BaseDataLoader` 要求当前 mode 使用的 batch size 能被可见设备数整除。改单卡可见性，或修改 `training.batch_size`、`sampling.batch_size`、`val.batch_size`、`test.batch_size` 中对应的一项。

Linux 示例：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config smoke_fdu_synpu_128.yaml --mode train --training_from_scratch
```

### 找不到 checkpoint 或加载 `None`

确认文件位于当前配置计算出的 `assets/<数据>/<模型或扩散>/ckpt/epoch_N.pth`。配置名称变了，输出目录也可能随之变化。用 `--sampling_from_epoch N` 时必须真实存在 `epoch_N.pth`。

### W&B 在等待登录

如果只想本地跑通，把配置改成：

```yaml
iio:
  use_tensorboard: true
  use_wandb: false
  save_pth_to_wandb: false
```

不要直接复用 `hyper_local_main.py` 中硬编码的本地凭据；应改用环境变量或自己的 W&B 登录信息。

### 普通模型出现 `KeyError: 'DLPU'` 等注册表错误

当前版本的 `selector/mmodel_selector.py` 按 `config.model.name` 查找包装器，但现有包装器注册名是 `DLPUMModel`、`PUNetMModel` 等，而 YAML 又把这些名字放在 `mmodel.name`。这两者目前不一致，所以普通模型路线的示例配置可能在创建包装器时失败。这是代码层面的已知问题，不是数据目录问题；需要让注册表读取 `config.mmodel.name`，或统一配置与注册名后再运行。

### `.sh` 脚本引用的配置不存在

根目录脚本更像作者的实验记录，部分命令仍引用已删除或移入历史目录的配置。以 `main.py --help`、当前 `configs/` 中真实存在的文件和本指南为准。Windows PowerShell 也不直接执行 Bash 的 `export` 语法。

## 11. 当前代码库的边界

- `test/` 被 `.gitignore` 忽略，当前仓库没有可直接运行的正式自动化测试套件；其中本地脚本主要用于画图、裁剪或统计数据。
- 配置和实验变体很多，但没有 schema 校验。字段拼错可能到运行较深的位置才报错。例如某配置里存在 `amp: fasle`，而当前单卡训练的 AMP 代码又被注释掉。
- CPU 不是当前训练主路径，多卡路径也需要按实际环境单独验证。
- `dockerfile` 提供 PyTorch 2.6/CUDA 12.6 基础环境，但工作目录名和 SSH 配置仍带有模板痕迹，使用前要审查，不要把真实密钥直接写进镜像。

这些边界不妨碍你阅读算法或复现实验，但意味着第一次上手应先做一轮最小配置试跑，再开始长时间训练。

## 12. 我应该从哪里开始改

- **换数据集**：先实现 `torch.utils.data.Dataset`，再在 `selector/data_selector.py` 注册加载器，最后让 `data.name` 对上注册名。
- **换网络**：在 `model/` 实现并用 `register_model` 注册，修改 `model.name`。如果走普通监督路线，还要检查对应 `MModel` 包装器。
- **换扩散过程**：在 `diffusion/` 实现并用 `register_diffusion` 注册，修改 `diffusion.name`，同时确认 `run/sample.py` 的结果保存分支支持它。
- **换损失**：在 `run/losses.py` 注册新损失，修改 `loss_type.name`。
- **换指标**：在 `meter/` 实现并注册，修改 `meter.name`。
- **只改实验参数**：复制 YAML 后修改，不要先动训练循环。

推荐的学习节奏是：先用一条样本检查数据字典，再用 1 个 epoch 跑通训练，接着完成一次采样和验证，最后才读 `FduDDPMDiffusion` 与 `FDUNet` 的内部实现。这样每读一层代码，都知道它的输入、输出和实际位置。

## 相关入口

- [项目快速使用说明](../readme.md)
- [主入口](../main.py)
- [已验证的单卡 smoke 配置](../configs/smoke_fdu_synpu_128.yaml)
- [主扩散实验配置](../configs/fdu_synpu_128_big.yaml)
- [扩散训练流程](../run/train.py)
- [数据加载器注册表](../selector/data_selector.py)
