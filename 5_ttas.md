请基于我当前项目代码实现一个新的“方向小波高频注意力残差重标定模块”，用于替换当前直接将方向高频特征作为 residual 注入主干的方式。

核心目标：
当前 Directional HF 模块采用类似

    Y = X + gamma * Hf

的方式直接将 Haar DWT 得到的方向高频特征注入主干。实验发现该方式在 x0-prediction 上有效，但在 epsilon-prediction 上可能引入额外高频扰动。

现在改为：

“方向高频不直接作为内容注入主干，而是作为 attention prior，对原始主干特征 X 进行增强/抑制重标定。”

即：

    Directional HF -> Attention Map -> Calibrate X -> Residual Output

==================================================
1. 输入输出
==================================================

输入：

    X: [B, C, H, W]

输出：

    Y: [B, C, H, W]

要求：
- 输入输出 shape 完全一致；
- 可以直接替换现有 HF residual 模块；
- 不修改 U-Net 其他部分；
- 不修改现有 diffusion training/inference pipeline；
- 同时兼容 x0-prediction 和 epsilon-prediction。

==================================================
2. Haar DWT
==================================================

首先对 X 做现有项目中的 Haar DWT：

    LL, LH, HL, HH = DWT(X)

其中：

    LL: low-frequency
    LH: directional high-frequency
    HL: directional high-frequency
    HH: diagonal high-frequency

本模块不使用 LL。

必须优先复用项目现有 DWT 实现，不要重新实现另一套 Haar DWT。

==================================================
3. Directional High-Frequency Encoder
==================================================

对三个高频子带分别进行独立方向卷积，不共享参数。

设计：

    F_lh = Conv_direction_lh(LH)
    F_hl = Conv_direction_hl(HL)
    F_hh = Conv_direction_hh(HH)

优先使用：

    LH -> Conv(1x3)
    HL -> Conv(3x1)
    HH -> Conv(3x3)

注意：
需要检查当前项目 Haar DWT 对 LH/HL 的定义。

如果代码中的 LH 表示垂直高频、HL 表示水平高频，则根据实际定义交换 1x3 和 3x1。

不要仅根据变量名称猜测，检查现有 DWT 实现。

每个方向卷积后可以使用：

    Conv -> GroupNorm -> SiLU

尽量保持轻量。

==================================================
4. 高频融合
==================================================

三个方向特征进行通道拼接：

    F_hf = concat([F_lh, F_hl, F_hh], dim=1)

如果每个分支输出 C 通道：

    F_hf: [B, 3C, H/2, W/2]

使用 1x1 Conv 进行融合和降维：

    F_fuse = Conv1x1(F_hf)

得到：

    F_fuse: [B, C, H/2, W/2]

这里不要使用 global self-attention / cross-attention。

==================================================
5. 生成 Directional HF Attention
==================================================

由方向高频融合特征生成 attention logits：

    A_logits = Conv1x1(F_fuse)

输出通道数为 C：

    A_logits: [B, C, H/2, W/2]

然后将其上采样到与 X 一致：

    A_logits = interpolate(
        A_logits,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

再：

    A = sigmoid(A_logits)

得到：

    A: [B, C, H, W]
    A ∈ [0, 1]

注意：
这里 A 是 channel-spatial attention，而不是一个单通道 mask。

==================================================
6. 中心化 Attention
==================================================

不要直接：

    Y = X * A

因为初始化 A≈0.5 会直接把 X 缩小一半。

改成：

    M = 2 * A - 1

因此：

    M ∈ [-1, 1]

含义：

    M > 0：增强
    M = 0：保持
    M < 0：抑制

==================================================
7. Residual Feature Calibration
==================================================

最终不要直接注入 F_hf。

禁止：

    Y = X + gamma * F_hf

改成：

    delta_X = alpha * X * M

    Y = X + delta_X

即：

    Y = X + alpha * X * (2 * A - 1)

等价于：

    Y = X * [1 + alpha * (2*A - 1)]

核心思想：

    高频只决定“原特征 X 应该在哪里/哪些通道增强或抑制”，
    高频特征本身不直接作为 residual 内容注入 X。

==================================================
8. alpha 设计
==================================================

alpha 设置为可学习参数。

第一版优先实现 scalar learnable alpha：

    self.alpha = nn.Parameter(...)

建议使用安全的小值初始化，例如：

    alpha_init = 0.1

并允许通过 config 修改。

如果项目已有 residual scale / gamma 参数设计，请尽量兼容现有配置，但不要同时使用两个功能重复的 scale。

最终：

    Y = X + alpha * X * M

==================================================
9. Identity-friendly 初始化
==================================================

Attention 最后一层 Conv 必须采用 zero initialization：

    weight = 0
    bias = 0

这样训练开始：

    A_logits = 0

因此：

    A = sigmoid(0) = 0.5

    M = 2*0.5 - 1 = 0

所以：

    Y = X

必须验证初始化后：

    max_abs(Y - X)

接近 0。

这样保证新模块在训练初始状态是严格/近似 identity，不会一开始破坏 diffusion U-Net 的主干特征。

==================================================
10. 禁止事项
==================================================

本版本不要：

1. 不要使用 LL；
2. 不要使用 Q/K/V Cross-Attention；
3. 不要直接把 LH/HL/HH 或 F_hf 加到 X；
4. 不要使用 Y = X * sigmoid(A)；
5. 不要堆叠大量卷积；
6. 不要修改现有 diffusion scheduler；
7. 不要修改现有 loss；
8. 不要修改训练 timestep；
9. 不要修改 inference timestep；
10. 不要改变其他训练超参数。

本次实验只验证：

“Directional HF residual injection”

改成

“Directional HF-guided residual calibration”

是否有效。

==================================================
11. 模块数学形式
==================================================

代码注释和文档中给出完整数学逻辑：

    {LL, LH, HL, HH} = DWT(X)

    F_LH = f_LH(LH)
    F_HL = f_HL(HL)
    F_HH = f_HH(HH)

    F_H = Concat(F_LH, F_HL, F_HH)

    F = f_fuse(F_H)

    A = sigmoid(
        Upsample(
            f_att(F)
        )
    )

    M = 2A - 1

    Y = X + alpha * X ⊙ M

其中：

    ⊙

表示逐元素乘法。

==================================================
12. 需要增加诊断统计
==================================================

训练/验证过程中增加可选 diagnostics，但不要影响反向传播。

至少统计：

1. attention mean

    mean(A)

2. attention std

    std(A)

3. centered attention mean

    mean(M)

4. centered attention std

    std(M)

5. alpha 当前值

6. residual ratio

    ||Y-X||_2 / (||X||_2 + eps)

7. positive calibration ratio

    ratio(M > 0)

8. negative calibration ratio

    ratio(M < 0)

这些数据用于判断模块是否真正学到了空间/通道重标定，而不是退化为常数缩放。

==================================================
13. 特别关注 epsilon prediction
==================================================

如果项目可以获得 diffusion timestep t，请额外支持按 timestep bucket 统计：

    attention_std(t)
    residual_ratio(t)
    mean(M)(t)

例如 T=200 时：

    0-39
    40-79
    80-119
    120-159
    160-199

这里只做统计，不要在第一版中使用 timestep 控制 alpha。

目的是分析：

    不同噪声阶段，Directional HF Attention 是否具有不同响应。

==================================================
14. 消融实验兼容
==================================================

请保留配置开关，使以下模型可以使用完全相同的训练代码：

A. baseline/off

    Y = X

B. directional_hf_residual

    Y = X + gamma * Hf

即当前方向高频直接残差版本。

C. directional_hf_attention

    Y = X + alpha * X * (2*A - 1)

即本次新版本。

确保 B 和 C 尽量使用相同的：

    Haar DWT
    directional conv
    HF fusion

二者主要区别应该是：

B：
    高频作为 residual content

C：
    高频作为 attention/calibration prior

不要同时修改其他结构，以保证消融公平。

==================================================
15. 单元测试
==================================================

实现后至少检查：

Test 1:
输入随机：

    X = torch.randn(B,C,H,W)

输出：

    Y.shape == X.shape

Test 2:
初始化状态：

    max(abs(Y-X)) < 1e-6

或在浮点误差允许范围内接近 0。

Test 3:
完成一次 backward：

    loss = Y.mean()
    loss.backward()

检查 directional conv、HF fusion、attention conv、alpha 是否能够正常参与梯度传播。

注意：
由于 attention 最后一层 zero-init，初始时前面的 HF encoder 在第一步可能因为最后一层权重为 0 而暂时没有梯度，这是正常现象。
必须确认 attention 输出层和 alpha/相关参数的梯度行为符合预期，并在后续优化 step 后 HF encoder 能获得梯度。

Test 4:
兼容当前使用的：
    fp16 / AMP
    CUDA
    batch training

==================================================
16. 最终输出
==================================================

完成后请：

1. 列出修改/新增的文件；
2. 给出新模块类名；
3. 给出 forward 数据流；
4. 给出关键公式；
5. 给出新增 config 参数；
6. 给出参数量变化；
7. 给出 FLOPs 或近似计算量变化（如果项目已有统计工具）；
8. 给出 identity initialization 测试结果；
9. 给出一次 forward/backward 测试结果；
10. 不要自动开始完整训练，等我确认后再运行正式实验。