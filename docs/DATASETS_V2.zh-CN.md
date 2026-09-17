# 第二版数据集：GFS、RME、RBR

旧数据目录予以保留，后续新实验不再使用。第二版实验使用 GFS、RME、RBR 三个数据族，三者均已生成。

## 名称与来源

- `GFS` 是本项目名称，对应 U3Net 论文中的 `MoGR`：Gaussian functions 的混合叠加随机斜坡，然后缩放相位范围。
- `RME` 对应 U3Net 论文的 Random Matrix Enlargement：从 2–10 阶随机方阵出发，随机选择均匀/正态分布和双线性/双三次插值，再缩放相位范围。
- `RBR` 是 Real-terrain Based Reconstruction：使用 Copernicus GLO-30 真实 DSM、随机 InSAR 成像几何、可控形变、空间变化相干性和复数域失相干噪声生成。完整协议见 `docs/RBR_DATASET.zh-CN.md`。

论文原始实验为 256×256。本项目按需求生成 128×128、64×64、32×32 三版。三个尺寸共享场景参数和训练 SNR 标签；每个尺寸都从干净相位重新加噪和缠绕，禁止直接缩放已经缠绕的图像。

## 数量与噪声

每个数据族、每个尺寸包含：

| 划分 | 数量 | SNR |
|---|---:|---|
| train | 5,000 | 每张从 0、5、10、20、30、60 dB 均匀抽取 |
| test_0dB | 1,000 | 固定 0 dB |
| test_5dB | 1,000 | 固定 5 dB |
| test_10dB | 1,000 | 固定 10 dB |
| test_20dB | 1,000 | 固定 20 dB |
| test_30dB | 1,000 | 固定 30 dB |

这里将论文“生成 1000 张测试图并在五个 SNR 上测试”落实为每个 SNR 1,000 张，与 U3Net 仓库按 `test data_10dB` 等独立文件读取的方式一致。测试集的 1,000 个场景 ID 在五个 SNR 和三个尺寸之间对应。

## 文件结构

```text
data/
├─ DATASETS_V2.json
├─ GFS128/              # GFS64、GFS32 同结构
│  ├─ train.h5
│  ├─ test_0dB.h5
│  ├─ test_5dB.h5
│  ├─ test_10dB.h5
│  ├─ test_20dB.h5
│  ├─ test_30dB.h5
│  └─ manifest.json
├─ RME128/              # RME64、RME32 同结构
└─ RBR128/              # RBR64、RBR32 同结构；含额外物理字段
```

每个 HDF5 文件含 `psi`（含噪缠绕相位）、`phi`（干净绝对相位）、`snr`、`scene_id`。RBR 额外包含 `phi_wrapped_clean`、`coherence`、`dem`、`deformation_phase`、`noise_map` 和成像几何/空间划分元数据。总清单记录形状、值域、有限性检查、文件大小和 SHA-256。

## 可复现生成

```powershell
.\.venv\Scripts\python.exe -B experiments/generate_u3_datasets.py
.\.venv\Scripts\python.exe -B -u experiments/generate_rbr_dataset.py --update-manifest
```

默认随机种子为 `20240913`。正式文件存在时，生成器通过临时文件写入后再原子替换，避免把中断文件当作完成数据。

项目数据加载器名称为 `GFS128/GFS64/GFS32`、`RME128/RME64/RME32` 和 `RBR128/RBR64/RBR32`。配置中的 `data.test_snr` 控制读取哪个测试噪声等级；默认 30 dB。

## 复现边界

U3Net 官方 GitHub 仓库发布了 HDF5 读取和训练代码，但没有发布 GFS/MoGR 或 RME 生成器。论文及其补充材料也没有给出 RME 随机分布的参数和插值坐标约定。因此本项目作如下可复现约定：Uniform 使用 `U(0,1)`，Normal 使用 `N(0,1)`，插值使用 Pillow 浮点图像的 Bilinear/Bicubic。GFS 的 Gaussian 与 ramp 参数范围取自论文引用的 SQD-LSTM 官方生成源码，并将其 0–255 坐标连续采样到目标尺寸。

因此，这些数据遵循 U3Net 论文公开的生成方法、数量、相位范围和噪声协议，但不能描述为与作者未公开的 HDF5 文件逐位相同。
