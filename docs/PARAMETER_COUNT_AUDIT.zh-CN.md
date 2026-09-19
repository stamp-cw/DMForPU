# 全项目方法参数量与原仓库核对

审计日期：2026-09-12。参数量必须在相同输入/输出通道、深度和选项下比较。PyTorch 统计 `numel()`；跨框架的 SQD-LSTM 以可训练参数为主口径。上游均固定到下表提交。

## 有明确原仓库的移植方法

| 方法 | 本地参数量 | 上游同配置参数量 | 结论 |
|---|---:|---:|---|
| DLPU | 26,114,881 | 26,114,881 | 一致；state dict 键和前向等价检查通过 |
| PUNet | 2,147,393 | 2,147,393 | **已修复，与官方结构一致** |
| SQD-LSTM | 895,473 可训练 | 895,473 可训练 | 可训练参数一致 |
| Restormer（1→1 通道） | 26,124,052 | 26,124,052 | 同配置一致 |
| Uformer-S（1→1、modulator） | 20,749,467 | 20,749,467 | 同配置一致 |
| U3Net（3 阶段） | 742,143 | 742,143 | 一致 |

### PUNet：已修复

原本地实现是根通道 8、深度 5 的 U-Net 编码器—解码器，只有 344,609 个参数。现已替换为公开 PUNet 仓库 `Wu-Patrick/Deformation-Monitoring-Dev` 的结构：64 通道、8 个 DilatedBlock、10 个 ResidualBlock 和 3×3 输出层，参数量恢复为 2,147,393。`experiments/results/full_200` 中已经完成的旧 PUNet 权重仍属于原 344,609 参数变体，不能加载到修复后的网络，也不能追溯改称官方 PUNet。

上游：<https://github.com/Wu-Patrick/Deformation-Monitoring-Dev/tree/5be3c0117e20742bf4f68dc828fc0399d1e362b4>

### SQD-LSTM：统计口径不同，但可训练参数一致

本地可训练参数为 895,473，与 Keras 上游完全相同。本地 `sum(p.numel() for p in model.parameters())` 得到 895,985，多出的 512 是 PyTorch LSTM 必须注册、但已冻结且保持为零的第二组 recurrent bias。Keras 的 `model.count_params()` 是 896,465，其中包括 895,473 个可训练参数和 992 个 BatchNorm moving mean/variance；PyTorch 将后者存为 buffer。因此跨框架应报告 895,473 个可训练参数。本项目先前报告的 895,985 是“已注册 Parameter 元素”，不是同口径的可训练参数。

上游：<https://github.com/Laknath1996/DeepPhaseUnwrap/blob/df8bad82cdbde376c3516e8980eca0e0e9f80f75/src/models/architectures.py>

### Restormer 与 Uformer：通道适配后的同配置一致

Restormer 上游默认 3→3 通道为 26,126,644；本项目相位回归使用 1→1 通道，在上游用相同通道实例化后为 26,124,052，与本地一致。

Uformer-S 上游默认 3→3 通道为 20,751,197；本项目使用 1→1 通道、`embed_dim=32`、九层深度均为 2、窗口 8、`modulator=True`，上游同选项为 20,749,467，与本地一致。通道适配是有意修改，不能拿本地数目直接和上游三通道默认数目比较。

上游：<https://github.com/swz30/Restormer/tree/68dc6ac472db26f16361150cb7a96a1bc87da93f>，<https://github.com/ZhendongWang6/Uformer/tree/65fc970a8ffc09605faca74ed016ee93c9ad8a36>

### DLPU 与 U3Net

DLPU 的网络权重键、形状和前向结果与官方一致，参数均为 26,114,881。U3Net 的 `UNet_SA.py` 与官方文件 SHA-256 完全一致；三阶段主干 724,212、条件网络 17,931，合计 742,143。

上游：<https://github.com/kqwang/Phase_unwrapping_by_U-Net/tree/3c84f34846dd8e9fbbdc613d161e278c2279d1bc>，<https://github.com/chenzhile1999/Unsupervised-PU/tree/47c0af31fee5ee7f442aa433a5fb3483a4376334>

## 本项目扩散方法与消融

这些方法没有一个可用于“参数量应完全相同”的外部原仓库模型。当前 HF 基线使用无交叉注意力的 Hugging Face `UNet2DModel`；WWFCA 变体使用含自定义频域交叉注意力的 `FDUNet`。通道、层数、DCC、WWFCA或物理展开模块由本项目设置，只能核对实现是否符合本项目配置：

| 本地方法/配置 | 参数量 | 与外部原仓库比较 |
|---|---:|---|
| HF small baseline | 1,046,017 | 无交叉注意力的 HF `UNet2DModel` |
| Chen-HF full | 1,046,973 | 无交叉注意力 HF 骨干加物理模块 |
| Chen-HF no-sparse | 1,046,378 | 本项目消融，无同构上游模型 |
| HF matched（四层 128） | 11,334,913 | 无交叉注意力的 HF 匹配规格实例 |
| DCC-only | 11,336,065 | 无交叉注意力；多 1,152 来自额外条件输入通道 |
| WWFCA-only | 13,908,481 | 使用修正后的 WWFCA 频域交叉注意力 |
| FDU（DCC+WWFCA） | 13,909,633 | 本项目方法，无外部同构原版 |
| FDUNetV4 对应配置 | 61,168,065 | 本地历史变体 |
| FDUNetV6 对应配置 | 238,001,921 | 本地历史变体 |

`FDUNetV2`、`FDUNetV5` 虽仍有注册类，但没有任何现行 YAML 选择它们；其参数量依赖必须另外指定的五层通道配置，不能给出唯一“项目配置参数量”。`AuxUNet`、`DiffAuxUNet` 是源代码中的辅助/备份组件，也没有现行实验配置，未作为独立方法计入。

Chen 等 2024 的 U3Net 是 Chen 论文的官方方法；`ChenHFDiffusion` 则是本项目基于论文思想新增的 HF diffusion，二者不是同一网络，参数量不应相等。

## 传统方法

Itoh、质量引导/MST、最小二乘等传统方法没有可训练神经网络参数，参数量记为 0；这不是移植参数量一致性的有效判断标准。

## 对现有实验的影响

1. DLPU、SQD-LSTM、Restormer、Uformer、U3Net 的参数量没有发现移植错误；SQD-LSTM 需要统一采用“可训练参数”口径。
2. **PUNet 结构已修复。** 已完成的旧 200 轮权重仍应标为 `PUNet-local`；论文需要的官方 PUNet 必须用修复后的 2,147,393 参数网络重新训练。
3. FDU、HF diffusion 和 Chen-HF 属于本项目配置或新方法，参数量应随配置完整披露，不应写成“与原仓库一致”。
