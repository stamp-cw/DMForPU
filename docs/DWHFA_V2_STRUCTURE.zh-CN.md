# DWHFA-v2网络结构

DWHFA-v2在原始DWHFA的基础上，将DWHFA下采样模块从一个阶段扩展到前三个UNet下采样阶段。主干、输入、扩散调度器和输出头保持不变。

## 插入位置

| 阶段 | 输入特征 | DWT子带尺寸 | 原生下采样 |
|---|---|---|---|
| Stage 0 | (B\times128\times128\times128) | (64\times64) | 128→64 |
| Stage 1 | (B\times128\times64\times64) | (32\times32) | 64→32 |
| Stage 2 | (B\times128\times32\times32) | (16\times16) | 32→16 |

每个阶段都独立执行：

\[
(LL,LH,HL,HH)=\operatorname{DWT}_{Haar}(X),
\]

丢弃 (LL)，分别对 (LH)、(HL)、(HH) 使用 (3\times1)、(1\times3)、(3\times3) 深度卷积和逐点卷积，然后通过 (1\times1) 融合卷积产生高频表示。注意力logits经过上采样和sigmoid得到 (A)，并构造：

\[
M=2A-1,\qquad Y=X+\alpha X\odot M.
\]

每个阶段拥有独立的高频编码器、校准卷积和可学习 (alpha)，且注意力卷积零初始化，所以三个阶段在初始化时都是恒等映射。

## 与原DWHFA的区别

- 原DWHFA：仅替换 `down_blocks[2].downsamplers[0]`；
- DWHFA-v2：替换 `down_blocks[0]`、`down_blocks[1]` 和 `down_blocks[2]` 的下采样器；
- 不使用Q/K/V交叉注意力；
- 不使用窗口划分；
- 不使用IDWT；
- 不使用LL作为查询，仅使用三个方向高频子带生成校准图。

实现位置：`model/fdunet/dwhfa.py::DWHFAv2UNet`，扩散注册位置：`diffusion/dcc_wwfca_diffusion.py`中的`dwhfa_v2`分支。

当前模型参数量：11,690,884（宽度配置`(128,128,128,128)`，每个块2个残差层）。
