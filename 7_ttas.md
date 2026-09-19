请阅读我当前 diffusion phase unwrapping 项目的代码。

我现在的数据集不是单一 wrapped phase。

对于每个 GT unwrapped phase x0：

x0
├── clean wrapped phase
├── 30 dB wrapped phase
├── 20 dB wrapped phase
├── 10 dB wrapped phase
├── 5 dB wrapped phase
└── 0 dB wrapped phase

这些 phase 全部对应同一个 GT。

==================================================
目标
==================================================

不要使用随机 re-corruption：

y_rc = W(y + u)

而是利用数据集天然提供的：

multi-SNR views

构建：

Multi-SNR Consistency Diffusion (MSCD)

训练框架。

==================================================
第一步：检查数据集结构
==================================================

首先检查当前 Dataset。

确认：

一个 sample 是否能够同时访问：

x0
y_clean
y_30
y_20
y_10
y_5
y_0

如果当前 Dataset 只能返回一个 SNR，

请修改 Dataset。

要求：

__getitem__(idx)

返回：

{
    "x0": ...,
    "clean": ...,
    "30db": ...,
    "20db": ...,
    "10db": ...,
    "5db": ...,
    "0db": ...
}

或者返回等价结构。

不要破坏现有数据增强。

==================================================
第二步：训练时随机选择两个 SNR
==================================================

定义：

High-SNR branch

Low-SNR branch

要求：

SNR_H > SNR_L

例如：

clean -> 20
30 -> 10
20 -> 5
10 -> 0

随机选择。

不要固定：

clean -> 0

每个 iteration 随机采样。

实现：

def sample_snr_pair(...):

返回：

condition_high
condition_low

并记录：

snr_high
snr_low

方便 logging。

==================================================
第三步：Diffusion 部分保持完全不变
==================================================

GT：

x0

随机：

t

随机：

epsilon

生成：

x_t =
    sqrt(alpha_bar_t) * x0
    + sqrt(1-alpha_bar_t) * epsilon

或者继续使用：

scheduler.add_noise()

注意：

整个 iteration 只生成一个：

x_t

不要为 high 和 low branch
分别生成不同 diffusion noise。

必须：

High branch：
    SAME x_t

Low branch：
    SAME x_t

SAME timestep
SAME epsilon

==================================================
第四步：High-SNR Branch
==================================================

epsilon_pred_H =
    model(
        x_t,
        t,
        condition_high
    )

恢复：

x0_pred_H

使用当前项目已有：

epsilon -> x0

转换方式。

计算：

L_eps_H =
    MSE(
        epsilon_pred_H,
        epsilon
    )

L_phase_H =
    L1(
        x0_pred_H,
        x0
    )

==================================================
第五步：Low-SNR Branch
==================================================

epsilon_pred_L =
    model(
        x_t,
        t,
        condition_low
    )

恢复：

x0_pred_L

计算：

L_eps_L =
    MSE(
        epsilon_pred_L,
        epsilon
    )

L_phase_L =
    L1(
        x0_pred_L,
        x0
    )

==================================================
第六步：Teacher / Student
==================================================

Teacher：

x0_pred_H.detach()

Student：

x0_pred_L

实现：

teacher_x0 =
    x0_pred_H.detach()

不要 detach student。

==================================================
第七步：Multi-SNR Consistency
==================================================

定义：

L_abs

L_grad

L_wrap

--------------------------------------------------
1. Absolute Consistency
--------------------------------------------------

L_abs =
    mean(
        abs(
            x0_pred_L
            -
            teacher_x0
        )
    )

--------------------------------------------------
2. Gradient Consistency
--------------------------------------------------

实现：

gradient_x
gradient_y

计算：

grad_H
grad_L

L_grad =
    0.5 * (
        L1(
            grad_x_L,
            grad_x_H
        )
        +
        L1(
            grad_y_L,
            grad_y_H
        )
    )

--------------------------------------------------
3. Wrapped Consistency
--------------------------------------------------

delta =
    x0_pred_L
    -
    teacher_x0

先转换到真实 phase domain。

然后：

wrapped_delta =
    atan2(
        sin(delta),
        cos(delta)
    )

L_wrap =
    mean(
        abs(
            wrapped_delta
        )
    )

==================================================
第八步：MSC Loss
==================================================

L_MSC =
    L_abs
    +
    lambda_grad * L_grad
    +
    lambda_wrap * L_wrap

默认：

lambda_grad = 0.5
lambda_wrap = 0.5

==================================================
第九步：最终 Loss
==================================================

定义：

L_total =
    L_eps_H
    +
    lambda_noise * L_eps_L
    +
    lambda_phase * (
        L_phase_H
        +
        L_phase_L
    )
    +
    lambda_msc * L_MSC

默认：

lambda_noise = 1.0
lambda_phase = 0.01
lambda_msc = 0.05
lambda_grad = 0.5
lambda_wrap = 0.5

即：

L_total =
    L_eps_H
    +
    1.0 * L_eps_L
    +
    0.01 * (
        L_phase_H
        +
        L_phase_L
    )
    +
    0.05 * (
        L_abs
        +
        0.5 * L_grad
        +
        0.5 * L_wrap
    )

==================================================
第十步：Curriculum Learning（可选开关）
==================================================

新增：

--use_curriculum

默认 False。

如果开启：

Stage 1:
    clean / 30 / 20

Stage 2:
    30 / 20 / 10

Stage 3:
    20 / 10 / 5 / 0

逐渐增加难度。

如果关闭：

所有 pair 均匀随机。

==================================================
第十一步：Ablation
==================================================

新增：

loss_type

支持：

baseline

phase

msc

--------------------------------------------------

baseline:

L = L_eps_H

--------------------------------------------------

phase:

L =
    L_eps_H
    +
    lambda_phase
    *
    L_phase_H

--------------------------------------------------

msc:

完整 MSCD

==================================================
第十二步：Logging
==================================================

记录：

loss_total

loss_eps_H
loss_eps_L

loss_phase_H
loss_phase_L

loss_abs
loss_grad
loss_wrap

loss_msc

snr_high
snr_low

==================================================
第十三步：Sanity Check
==================================================

测试：

High=Low=clean

理论上：

L_abs ≈ 0

L_grad ≈ 0

L_wrap ≈ 0

测试：

clean -> 0 dB

MSC loss 应显著增加。

确认：

teacher 无梯度

student 保留梯度

第二个 branch
正确参与反向传播。

==================================================
第十四步：代码要求
==================================================

不要修改：

UNet

DCC

WWFCA

scheduler

inference

checkpoint

只修改：

Dataset
Loss
Training Loop

保持旧 checkpoint 尽量兼容。

==================================================
最后输出
==================================================

请报告：

1. Dataset 如何修改；
2. 如何选择 SNR pair；
3. 修改了哪些文件；
4. 新增哪些 config；
5. 最终数学公式；
6. Teacher / Student 实现位置；
7. MSC 实现代码；
8. 是否发现 shape 问题；
9. 是否发现 normalization 问题；
10. 是否保持 baseline 可运行；
11. 训练 FLOPs 增加多少；
12. 推理 FLOPs 是否增加。