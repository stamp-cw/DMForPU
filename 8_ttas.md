请阅读我当前的 diffusion phase unwrapping 项目，并实现一个新的实验性 Loss：

Phase-Equivalent Adaptive Loss（PEA Loss）

目的：

当前模型使用 epsilon-prediction：

    epsilon_pred = model(x_t, timestep, condition)

目前 baseline loss 为：

    L_eps = MSE(epsilon_pred, epsilon)

我还测试过：

    L = L_eps + 0.01 * L1(x0_pred, x0)

但最终 phase reconstruction 精度仍明显低于直接 x0-prediction。

本实验不要修改 prediction_type。

模型仍然必须预测 epsilon。

本实验希望利用 epsilon prediction error 到 x0 reconstruction error 的数学关系，
设计 timestep-aware loss，使 epsilon-prediction 的训练目标更加面向最终 x0 reconstruction。

==================================================
1. 严格限制修改范围
==================================================

不要修改：

- UNet / diffusion backbone
- condition encoder
- DCC
- WWFCA
- Dataset
- DataLoader
- data augmentation
- noise scheduler
- beta schedule
- diffusion timestep 数量
- inference / sampling
- optimizer（除非 loss 配置需要）
- checkpoint model architecture
- validation metrics

只允许修改：

- loss implementation
- training loop 中调用 loss 的部分
- config / argparse
- logging
- 必要的 loss helper functions

目标是进行严格公平的 loss ablation。

==================================================
2. 首先检查当前项目
==================================================

修改前先检查并在最终报告中告诉我：

1. 当前 prediction_type 是否确实为 epsilon。

2. 当前 scheduler 类型。

3. 当前 scheduler 是否提供：

    alphas_cumprod

4. x0 的 normalization 范围。

例如：

    [-1,1]

或者其他范围。

5. condition 的 normalization。

6. 当前 timestep 范围，例如：

    0 ~ 199

7. 当前 epsilon baseline loss 的具体代码位置。

8. 当前：

    epsilon_pred -> x0_pred

是否已经存在 helper function。

如果存在，优先复用。

不要重复实现多个版本。

如果当前 prediction_type 不是 epsilon，
不要强行修改，先在最终报告中说明。

==================================================
3. 数学基础
==================================================

当前 forward diffusion：

    x_t =
        sqrt(alpha_bar_t) * x0
        +
        sqrt(1 - alpha_bar_t) * epsilon

定义：

    alpha_t = sqrt(alpha_bar_t)

    sigma_t = sqrt(1 - alpha_bar_t)

因此：

    x_t =
        alpha_t * x0
        +
        sigma_t * epsilon

网络：

    epsilon_pred =
        epsilon_theta(
            x_t,
            timestep,
            condition
        )

从 epsilon_pred 恢复 x0：

    x0_pred =
        (
            x_t
            -
            sigma_t * epsilon_pred
        )
        /
        alpha_t

即：

    x0_pred =
        (
            x_t
            -
            sqrt(1-alpha_bar_t) * epsilon_pred
        )
        /
        sqrt(alpha_bar_t)

必须保持这个过程可微。

绝对禁止：

    x0_pred.detach()

因为 phase reconstruction loss 必须通过：

    x0_pred
        ->
    epsilon_pred
        ->
    model parameters

反向传播。

==================================================
4. epsilon error -> x0 error
==================================================

数学上：

    x0_pred - x0
        =
    -(sigma_t / alpha_t)
    * (epsilon_pred - epsilon)

因此：

    ||x0_pred - x0||_2^2

等价于：

    (sigma_t^2 / alpha_t^2)
    *
    ||epsilon_pred - epsilon||_2^2

而：

    sigma_t^2 / alpha_t^2

等于：

    (1-alpha_bar_t) / alpha_bar_t

也等于：

    1 / SNR(t)

其中：

    SNR(t) =
        alpha_bar_t
        /
        (1-alpha_bar_t)

基于这个关系实现 timestep-aware epsilon weighting。

==================================================
5. 实现 Phase-Equivalent Ratio
==================================================

定义：

    r_t =
        (1 - alpha_bar_t)
        /
        (alpha_bar_t + eps)

其中：

    eps = 1e-8

注意：

这里的 SNR(t) 是 diffusion timestep SNR。

不是数据集中的：

    0 dB
    5 dB
    10 dB
    ...

measurement SNR。

不要混淆两者。

==================================================
6. 不直接使用 r_t
==================================================

直接：

    r_t * L_epsilon

在高噪声 timestep 下可能产生非常大的权重。

因此实现 bounded phase-equivalent weighting：

    w_bpe(t) =
        r_t
        /
        (1 + r_t / gamma)

其中：

    gamma > 0

默认：

    gamma = 5.0

性质：

当：

    r_t << gamma

有：

    w_bpe ≈ r_t

当：

    r_t -> infinity

有：

    w_bpe -> gamma

因此可以避免高 timestep 权重爆炸。

实现 helper：

    def compute_bpe_weight(
        timesteps,
        noise_scheduler,
        gamma=5.0,
        eps=1e-8,
    ):
        ...

返回：

    [B]

每个 batch sample 有自己的 timestep weight。

==================================================
7. epsilon loss 必须 per-sample 计算
==================================================

这一点非常重要。

禁止直接：

    F.mse_loss(
        epsilon_pred,
        epsilon
    )

得到一个 batch scalar 后再乘平均 weight。

因为一个 batch 中不同 sample 的 timestep 不同。

必须：

    mse_per_pixel =
        (epsilon_pred - epsilon) ** 2

然后：

    loss_eps_per_sample =
        mse_per_pixel.mean(
            dim=(1,2,3)
        )

得到：

    [B]

然后：

    loss_bpe =
        (
            w_bpe
            *
            loss_eps_per_sample
        ).mean()

即：

    L_BPE =
        E_batch[
            w_bpe(t)
            *
            MSE_sample(
                epsilon_pred,
                epsilon
            )
        ]

==================================================
8. epsilon -> x0 reconstruction
==================================================

使用：

    alpha_bar_t =
        noise_scheduler.alphas_cumprod[timesteps]

reshape：

    [B]
        ->
    [B,1,1,1]

然后：

    alpha =
        sqrt(alpha_bar_t)

    sigma =
        sqrt(1-alpha_bar_t)

恢复：

    x0_pred =
        (
            x_t
            -
            sigma * epsilon_pred
        )
        /
        alpha

需要 numerical safety。

例如：

    alpha =
        torch.clamp(
            alpha,
            min=1e-6
        )

但不要随意改变 scheduler 的 alpha_bar_t 本身。

如果项目已有 scheduler helper / predict_original_sample，
优先确认其公式完全一致后复用。

==================================================
9. Phase Reconstruction Loss
==================================================

第一版实验只使用最简单的：

    L_phase =
        L1(
            x0_pred,
            x0
        )

不要加入：

- gradient loss
- wrap loss
- circular loss
- SSIM loss
- perceptual loss
- frequency loss
- Multi-SNR consistency
- HRPC
- teacher/student
- re-corruption

本实验只验证：

    timestep-aware epsilon weighting
        +
    timestep-aware x0 supervision

==================================================
10. Phase Reliability Weight
==================================================

由于：

    x0_pred =
        (
            x_t
            -
            sigma_t * epsilon_pred
        )
        /
        alpha_t

在高噪声 timestep：

    alpha_t 很小

x0 reconstruction 对 epsilon error 非常敏感。

因此 phase loss 不应该在所有 timestep 使用完全相同的权重。

定义：

    q(t) =
        alpha_bar_t ** rho

默认：

    rho = 1.0

因此：

    q(t) = alpha_bar_t

性质：

低噪声 timestep：

    alpha_bar_t -> 1

因此：

    q(t) -> 1

phase reconstruction supervision 较强。

高噪声 timestep：

    alpha_bar_t -> 0

因此：

    q(t) -> 0

降低不可靠 x0_pred 对训练的影响。

==================================================
11. Phase Loss 也必须 per-sample
==================================================

不要直接：

    F.l1_loss(x0_pred, x0)

得到 scalar。

必须：

    phase_error =
        abs(
            x0_pred - x0
        )

    loss_phase_per_sample =
        phase_error.mean(
            dim=(1,2,3)
        )

得到：

    [B]

然后：

    loss_phase_weighted =
        (
            q_t
            *
            loss_phase_per_sample
        ).mean()

其中：

    q_t =
        alpha_bar_t ** rho

==================================================
12. 最终 PEA Loss
==================================================

最终：

    L_total =
        L_BPE
        +
        lambda_phase
        *
        L_phase_weighted

即数学形式：

    L_PEA =
        E[
            w_BPE(t)
            *
            ||epsilon_pred - epsilon||_2^2
        ]

        +

        lambda_phase
        *
        E[
            alpha_bar_t^rho
            *
            ||x0_pred - x0||_1
        ]

其中：

    r_t =
        (1-alpha_bar_t)
        /
        (alpha_bar_t + eps)

    w_BPE(t) =
        r_t
        /
        (1 + r_t / gamma)

默认参数：

    gamma = 5.0
    rho = 1.0
    lambda_phase = 0.01

==================================================
13. 一个重要实验问题：
BPE weight 的整体尺度
==================================================

请检查：

    mean(w_BPE)

因为 BPE weighting 会改变原始 epsilon loss 的整体 scale。

因此实现一个可选参数：

    normalize_bpe_weight

默认：

    False

如果：

    normalize_bpe_weight = True

则：

    w_bpe =
        w_bpe
        /
        (
            w_bpe.mean().detach()
            + 1e-8
        )

这样：

    mean(w_bpe) ≈ 1

可以避免 loss scale 改变导致实验不公平。

但是：

不要默认开启。

我要分别实验：

    raw BPE

和：

    normalized BPE

请实现配置开关。

==================================================
14. 保留所有 Baseline
==================================================

不要删除当前 loss。

增加：

    loss_type

至少支持：

----------------------------------
A. epsilon
----------------------------------

    L =
        MSE(
            epsilon_pred,
            epsilon
        )

完全保持当前 baseline。

----------------------------------
B. epsilon_phase
----------------------------------

当前已有实验：

    L =
        MSE(
            epsilon_pred,
            epsilon
        )
        +
        lambda_phase
        *
        L1(
            x0_pred,
            x0
        )

这里不要使用 timestep weighting。

用于复现当前：

    epsilon + 0.01 x0

结果。

----------------------------------
C. bpe
----------------------------------

只使用：

    L =
        L_BPE

用于判断：

    phase-equivalent epsilon weighting

本身有没有效果。

----------------------------------
D. pea
----------------------------------

完整：

    L =
        L_BPE
        +
        lambda_phase
        *
        L_phase_weighted

这是本次核心实验。

==================================================
15. Config 参数
==================================================

新增：

    --loss_type

支持：

    epsilon
    epsilon_phase
    bpe
    pea

新增：

    --lambda_phase

默认：

    0.01

新增：

    --bpe_gamma

默认：

    5.0

新增：

    --phase_rho

默认：

    1.0

新增：

    --normalize_bpe_weight

默认：

    false

所有参数不要硬编码到 training loop。

==================================================
16. Logging
==================================================

如果当前项目使用 W&B，
沿用已有 W&B，不要重新初始化。

至少记录：

    train/loss_total

    train/loss_epsilon_raw

    train/loss_bpe

    train/loss_phase_raw

    train/loss_phase_weighted

    train/bpe_weight_mean

    train/bpe_weight_min

    train/bpe_weight_max

    train/phase_weight_mean

另外记录：

    train/x0_pred_mae

其中：

    x0_pred_mae =
        mean(
            abs(
                x0_pred.detach()
                -
                x0
            )
        )

仅用于 logging，
这里 detach 没问题。

==================================================
17. timestep 分段统计
==================================================

为了验证这个 Loss 是否真的缩小 epsilon prediction
到 x0 prediction 的误差，请增加 validation 统计。

根据当前 T 自动划分 timestep 区间。

如果：

    T = 200

可以例如：

    [0, 40)
    [40, 80)
    [80, 120)
    [120, 160)
    [160, 200)

分别统计：

    x0 reconstruction MAE

即：

    val/x0_mae_t_0_40
    val/x0_mae_t_40_80
    val/x0_mae_t_80_120
    val/x0_mae_t_120_160
    val/x0_mae_t_160_200

如果当前 T 不是 200，
自动等分为 5 个区间。

不要因此改变 validation sampling。

==================================================
18. Numerical Stability
==================================================

重点检查：

1.

    alpha_bar_t

是否可能接近 0。

2.

    r_t

是否出现 Inf。

3.

    w_bpe

是否出现 NaN。

4.

    x0_pred

是否因为除以：

    sqrt(alpha_bar_t)

出现 Inf / NaN。

5.

AMP / FP16 情况下，
建议在计算：

    alpha_bar_t
    r_t
    w_bpe
    x0 reconstruction coefficients

时根据当前项目情况考虑使用 float32，
然后保持最终 loss 与 autograd 正确连接。

不要因为数值稳定处理而 detach epsilon_pred。

==================================================
19. Gradient Sanity Check
==================================================

请确认：

    epsilon_pred.requires_grad == True

    x0_pred.requires_grad == True

完整 PEA：

    loss_total.backward()

之后 model parameter 存在正常梯度。

特别确认：

    L_phase_weighted

单独 backward 时，

能够通过：

    L_phase
        ->
    x0_pred
        ->
    epsilon_pred
        ->
    UNet

产生梯度。

==================================================
20. 数学一致性测试
==================================================

增加一个简单 unit/sanity test。

随机生成：

    x0
    epsilon
    alpha_bar

构造：

    x_t =
        sqrt(alpha_bar) * x0
        +
        sqrt(1-alpha_bar) * epsilon

人为构造：

    epsilon_pred

然后计算：

    x0_pred

验证：

    x0_pred - x0

应近似等于：

    -sqrt(
        (1-alpha_bar)
        /
        alpha_bar
    )
    *
    (epsilon_pred - epsilon)

允许浮点误差。

另外验证 MSE：

    MSE(x0_pred,x0)

应近似：

    ((1-alpha_bar)/alpha_bar)
    *
    MSE(epsilon_pred,epsilon)

这是本 Loss 的数学基础。

==================================================
21. 不要修改 inference
==================================================

非常重要：

推理仍然使用标准 epsilon-prediction。

即：

    x_t
        ->
    model
        ->
    epsilon_pred
        ->
    scheduler.step(...)
        ->
    x_{t-1}

不要在 inference 中加入：

    BPE
    PEA
    x0 loss
    timestep loss weighting

这些全部只属于 training。

最终模型 prediction_type 仍然是：

    epsilon

==================================================
22. 第一轮实验不要加入其他创新
==================================================

本次修改完成后，不要擅自加入：

- v-prediction
- x0-prediction
- Min-SNR
- P2 weighting
- SNR gamma weighting 的其他版本
- gradient loss
- wrap consistency
- circular loss
- multi-SNR
- teacher/student
- re-corruption
- self-distillation
- frequency loss
- perceptual loss

我现在只需要验证：

Phase-Equivalent Adaptive Loss

是否能够缩小：

epsilon-prediction

与：

direct x0-prediction

之间的 phase reconstruction accuracy gap。

==================================================
23. 最终实验建议配置
==================================================

修改完成后，给我能够直接运行的四组配置/命令：

Experiment A:

    loss_type = epsilon

Experiment B:

    loss_type = epsilon_phase
    lambda_phase = 0.01

Experiment C:

    loss_type = bpe
    bpe_gamma = 5.0

Experiment D:

    loss_type = pea
    bpe_gamma = 5.0
    phase_rho = 1.0
    lambda_phase = 0.01

除此之外所有：

    seed
    dataset
    batch size
    epochs
    learning rate
    optimizer
    scheduler
    diffusion T
    inference steps
    augmentation

必须完全相同。

==================================================
24. 最后给我完整报告
==================================================

修改完成后请报告：

1. 修改了哪些文件。

2. 修改了哪些函数。

3. 当前 prediction_type。

4. 当前 diffusion timestep T。

5. 当前 x0 normalization。

6. 原 baseline loss 代码。

7. epsilon -> x0 的代码。

8. r_t 的代码。

9. BPE weight 的代码。

10. phase reliability q(t) 的代码。

11. 最终 PEA loss 的代码。

12. 每个 loss tensor 的 shape。

13. 是否正确进行 per-sample weighting。

14. 是否发现 NaN / Inf。

15. AMP 下是否安全。

16. phase loss 是否能够反向传播到 epsilon predictor。

17. inference 是否完全没有修改。

18. 给出 A/B/C/D 四组实验的直接运行命令。

19. 如果发现我的项目结构与上述假设不一致，
    不要强行套公式，
    根据实际代码进行最小适配并明确说明。