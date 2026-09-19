# DWHFA扩散解缠网络结构

本文档给出当前代码中 **Directional Wavelet High-Frequency Attention (DWHFA)** 的完整网络结构，可直接用于SCI论文的方法与网络结构图部分。结构图采用可编辑SVG绘制，文件为 [DWHFA_NETWORK_ARCHITECTURE.svg](DWHFA_NETWORK_ARCHITECTURE.svg)。

## 1. 总体结构

DWHFA以标准HF扩散UNet为主干，在倒数第二个下采样层（32×32特征到16×16特征）之前插入一个方向高频校准模块。给定带噪扩散状态 (x_t) 和包裹相位 (psi)，输入首先构造为

\[
z_t=\operatorname{Concat}\left(x_t,\frac{\psi}{\pi}\right)\in\mathbb{R}^{B\times2\times128\times128}.
\]

其中 (x_t) 和条件包裹相位使用项目现有的归一化方式。该模型使用 `diffusers.UNet2DModel`，4个下采样块和4个上采样块，每个块包含2个残差层，通道宽度为 `(128,128,128,128)`，不使用UNet内部的空间注意力或交叉注意力。DWHFA只改变倒数第二个下采样层的输入特征，主干的其他层保持不变。

## 2. DWHFA插入位置

设进入倒数第二个下采样层的特征为

\[
X\in\mathbb{R}^{B\times C\times32\times32},\qquad C=128.
\]

原始下采样算子记作 (D(\cdot))。DWHFA先对 (X)进行方向小波分析并得到校准特征 (Y)，然后执行原始下采样：

\[
Y=\operatorname{DWHFA}(X),\qquad X_{16}=D(Y).
\]

因此，DWHFA不是独立的输出分支，也不替换UNet的解码器；它是主干中一个保持空间尺寸的特征校准模块。

## 3. Haar方向高频分解

模块对 (X)执行单层二维Haar DWT：

\[
(LL,LH,HL,HH)=\operatorname{DWT}_{Haar}(X),
\]

四个子带的尺寸均为 (B\times C\times16\times16)。当前代码约定中：

- (LH)：垂直方向高频，使用 (3\times1) 深度卷积；
- (HL)：水平方向高频，使用 (1\times3) 深度卷积；
- (HH)：对角方向高频，使用 (3\times3) 深度卷积；
- (LL)：不进入DWHFA高频编码分支。

每个方向分支均由以下顺序组成：

\[
H_b=\operatorname{SiLU}(\operatorname{GN}(\operatorname{PWConv}(\operatorname{DWConv}_{k_b}(b)))),
\]

其中 (b\in\{LH,HL,HH\})，`DWConv`为逐通道卷积，`PWConv`为 (1\times1)逐点卷积，`GN`为GroupNorm。

三个方向特征随后拼接并融合：

\[
H=\operatorname{SiLU}\left(\operatorname{GN}\left(\operatorname{Conv}_{1\times1}
([H_{LH},H_{HL},H_{HH}])\right)\right)
\in\mathbb{R}^{B\times C\times16\times16}.
\]

## 4. 通道-空间校准与残差注入

融合后的高频特征经过一个 (1\times1)卷积产生注意力logits，并双线性上采样到原始特征的空间分辨率：

\[
A=\sigma\left(\operatorname{Interp}_{32\times32}
\left(\operatorname{Conv}_{1\times1}(H)\right)\right),
\quad A\in(0,1)^{B\times C\times32\times32}.
\]

代码中该 (1\times1)卷积的权重和偏置均零初始化，因此初始状态满足 (A=0.5)。将注意力转换为零均值调制图：

\[
M=2A-1.
\]

最终校准采用逐元素乘法残差注入：

\[
Y=X+\alpha\,(X\odot M),
\]

其中 (alpha) 是可学习标量，初始值为0.1。由于 (A=0.5) 时 (M=0)，模块初始近似恒等映射，不会在训练开始时随机破坏HF基线的特征分布。DWHFA不把高频特征作为新的内容直接相加，而是利用高频信息对原始特征 (X)进行通道-空间幅度校准。

## 5. 扩散预测头

校准特征经过原始UNet下采样、瓶颈和上采样路径后，由 (1\times1)输出卷积得到单通道预测结果。epsilon实验中，输出为噪声预测：

\[
\hat\epsilon=\epsilon_\theta(z_t,t,\psi).
\]

训练和推理仍使用原始DDPM scheduler；DWHFA只作用于网络特征提取阶段，不改变扩散时间步、噪声调度或采样公式。

## 6. 结构特点与论文表述

可在论文中将该模块概括为：

> DWHFA performs a single-level Haar decomposition on the penultimate UNet feature. The low-frequency subband is discarded, while vertical, horizontal, and diagonal high-frequency subbands are encoded by direction-specific depthwise-pointwise convolutions. The resulting high-frequency representation is fused to produce a channel-spatial calibration map. A zero-centered multiplicative residual is then injected into the original feature before the native UNet downsampling operation. Zero initialization of the attention logits makes the module identity-preserving at initialization.

需要明确区分以下几点：

1. 当前DWHFA不使用Q/K/V交叉注意力；
2. 当前DWHFA不使用LL查询高频，也不进行窗口划分；
3. 当前DWHFA不使用IDWT重建图像或特征；
4. 高频信息通过 (Y=X+\alpha X\odot(2A-1)) 调制原始特征；
5. DWHFA插入的是UNet倒数第二个下采样层之前，而不是输出端。

## 7. 与代码的对应关系

| 论文结构 | 代码位置 |
|---|---|
| DWHFA模块 | `model/fdunet/dwhfa.py::DWHFADownsample` |
| DWHFA封装UNet | `model/fdunet/dwhfa.py::DWHFAUNet` |
| 插入倒数第二个下采样层 | `DWHFAUNet.__init__` 中 `down_blocks[-2].downsamplers[0]` |
| Haar DWT | `model/fdunet/fdunet_atten.py::_haar_dwt2` |
| 扩散封装 | `diffusion/dcc_wwfca_diffusion.py` 中 `variant == "dwhfa"` |
| epsilon/x0训练目标 | `experiments/train_gfs128_t200_prediction_study.py` |

