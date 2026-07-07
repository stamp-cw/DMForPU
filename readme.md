# DMForPU 使用教程

这是一个论文代码仓库，入口统一走 `main.py`，通过 `--config` 选择配置文件，通过 `--mode` 选择训练、验证、采样或测试流程。

## 1. 环境安装

先安装 PyTorch 和基础依赖：

```bash
pip install -r requirements.txt
```

如果还缺包，再按 `install.sh` 里的列表补装。Windows 下直接执行 `python main.py ...` 即可，不必依赖 `.sh` 脚本。

## 2. 数据准备

配置里的 `data.name` 必须和 `selector/data_selector.py` 里注册的数据集名称一致。常见数据目录如下：

### SyntheticPUMat 系列

```text
data/<dataset_name>/
  train_in/*.mat
  train_gt/*.mat
  test_in/*.mat
  test_gt/*.mat
url: https://github.com/kqwang/Phase_unwrapping_by_U-Net
```

`.mat` 文件里通常读取 `input` 和 `gt` 字段。

### InSARDLPUMat 系列

```text
data/<dataset_name>/
  train_wrapped/*.mat
  train_absolute/*.mat
  test_wrapped/*.mat
  test_absolute/*.mat
url: https://github.com/zhoulifan/InSAR-DLPU
```

`.mat` 文件里通常读取 `input` 和 `output` 字段。

## 3. 最常用命令

注意：`--config` 只填 `configs/` 目录下的文件名，不要再加 `configs/` 前缀。

### 3.1 训练扩散模型

```bash
python main.py --config fdu_synpu_128_big.yaml --mode train --training_from_scratch
```

### 3.2 训练普通模型

```bash
python main.py --config dlpu_dlpu_256_big.yaml --mode train_model --training_from_scratch
```

### 3.3 采样

```bash
python main.py --config fdu_synpu_128_big.yaml --mode sample --sampling_from_epoch 100
```

### 3.4 验证

```bash
python main.py --config fdu_synpu_128_big.yaml --mode val --sampling_from_epoch 100
```

### 3.5 普通模型采样 / 验证

```bash
python main.py --config dlpu_dlpu_256_big.yaml --mode sample_model --sampling_from_epoch 100
python main.py --config dlpu_dlpu_256_big.yaml --mode val_model --sampling_from_epoch 100
```

## 4. 模式说明

| mode | 说明 |
| --- | --- |
| `train` | 训练扩散模型 |
| `sample` | 扩散模型推理采样 |
| `val` | 扩散模型验证 |
| `test` | 扩散模型测试 |
| `train_model` | 训练普通模型 |
| `sample_model` | 普通模型采样 |
| `val_model` | 普通模型验证 |
| `test_model` | 普通模型测试 |
| `train_multi` | 扩散模型多卡训练 |
| `train_multi_model` | 普通模型多卡训练 |

## 5. 常用参数

- `--training_from_scratch`：强制从头训练，不自动续训。
- `--sampling_from_epoch N`：指定加载第 `N` 轮 checkpoint；不传则默认用最新 checkpoint。
- `--hyper`：启用超参搜索流程。

## 6. 输出目录

训练和推理结果默认写到：

```text
assets/<data_name>/<diffusion_name or model_name>/
```

常见子目录：

- `ckpt/`：模型权重
- `sample/<epoch>/`：采样结果
- `val/<epoch>/`：验证结果
- `tb/`：TensorBoard 日志
- `wandb/`：wandb 本地缓存

## 7. 配置建议

- `training.batch_size`、`val.batch_size`、`sampling.batch_size` 要和 GPU 数量匹配。
- 训练时如果已经存在 `assets/.../ckpt/epoch_*.pth`，默认会自动续训。
- 如果想切换模型或数据集，优先改 `configs/*.yaml`，不要直接改训练代码。

## 8. 示例

```bash
python main.py --config wav_synpu_128_mid.yaml --mode train --training_from_scratch
python main.py --config wav_synpu_128_mid.yaml --mode sample --sampling_from_epoch 100
python main.py --config wav_synpu_128_mid.yaml --mode val --sampling_from_epoch 100
```
