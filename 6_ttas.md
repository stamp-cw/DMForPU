请阅读我当前项目中 diffusion phase unwrapping 的训练代码，在现有 epsilon-prediction
训练框架上实现一个新的损失函数：

Hierarchical Re-corruption Phase Consistency Loss (HRPC Loss)

目标：
借鉴 U3Net / “Unsupervised Deep Unrolling Networks for Phase Unwrapping”
中的 re-corruption 和 self-distillation 思想，将其改造成适用于 conditional diffusion
phase unwrapping 的监督训练损失。

重要：
这不是直接复现 U3Net Loss。
不要修改网络结构。
不要修改 DCC、WWFCA、UNet、dataset、scheduler 和 inference。
只修改训练阶段的 loss 和必要的辅助函数。

==================================================
0. 首先检查当前项目
==================================================

在修改代码前，请先检查：

1. 当前 diffusion prediction_type 是什么：
   - epsilon
   - sample/x0
   - v_prediction

2. 当前模型 forward 的输入：
   model(x_t, t, condition)
   分别对应什么 Tensor。

3. 确认：
   x0 = ground-truth unwrapped phase
   condition = wrapped phase
   x_t = diffusion noised x0

4. 确认 x0 的归一化范围。

5. 确认 wrapped phase condition 的归一化方式。
   特别注意：
   如果 condition 已经从 [-pi, pi] 归一化到 [-1,1]，
   wrapping 操作不能直接在 [-1,1] 上使用 2*pi 周期，
   必须先转换到物理相位域或实现与归一化尺度一致的 wrap operator。

6. 确认 noise_scheduler 是否为 HuggingFace diffusers scheduler，
   并检查：
   noise_scheduler.alphas_cumprod

7. 不要假设变量名。
   根据当前项目实际变量名完成适配。

如果当前不是 epsilon prediction，
先说明情况，不要直接套用下面的 epsilon -> x0 公式。

==================================================
1. Baseline diffusion noise loss
==================================================

保留当前 diffusion 原始训练目标：

epsilon_pred = model(x_t, t, condition)

L_eps = MSE(epsilon_pred, epsilon)

其中：

epsilon = forward diffusion 时真实加入的 Gaussian noise。

不要删除或改变原来的 diffusion noise objective。

==================================================
2. epsilon -> x0 reconstruction
==================================================

对于 epsilon prediction：

x_t =
    sqrt(alpha_bar_t) * x0
    + sqrt(1-alpha_bar_t) * epsilon

因此：

x0_pred =
    (x_t
     - sqrt(1-alpha_bar_t) * epsilon_pred)
    / sqrt(alpha_bar_t)

alpha_bar_t 必须直接从当前 scheduler 获取：

alpha_bar_t =
    noise_scheduler.alphas_cumprod[timesteps]

然后 reshape：

[B] -> [B,1,1,1]

注意：

- device 与 x_t 一致
- dtype 与 x_t 一致
- clamp 防止除零
- AMP/fp16 下避免 NaN/Inf
- 不允许 detach x0_pred

建议封装：

def predict_x0_from_epsilon(
    x_t,
    epsilon_pred,
    timesteps,
    noise_scheduler,
):
    ...

==================================================
3. Phase Reconstruction Loss
==================================================

增加直接的 absolute phase reconstruction：

L_phase =
    mean(abs(x0_pred - x0))

即：

L_phase = L1(x0_pred, x0)

默认：

lambda_phase = 0.01

注意：

x0_pred 和 x0 必须处于完全相同的归一化尺度。

==================================================
4. Condition Re-corruption
==================================================

现在实现 HRPC 的核心。

不要重新污染 x_t。

x_t 在 clean branch 和 re-corruption branch 中必须完全相同。

只重新污染 wrapped-phase condition。

原始：

condition = y

随机生成一个额外的小扰动：

u ~ N(0, sigma_rc^2)

然后：

condition_rc = Wrap(condition + u)

但是必须正确处理 condition 的单位。

如果 condition 是物理 wrapped phase：

condition ∈ [-pi, pi]

则：

Wrap(theta) =
    atan2(sin(theta), cos(theta))

推荐实现：

def wrap_phase(theta):
    return torch.atan2(torch.sin(theta), torch.cos(theta))

不要使用：

(theta + pi) % (2*pi) - pi

作为首选实现，因为 atan2(sin,cos) 对 autograd 更友好。

--------------------------------------------------
如果 condition 被归一化：
--------------------------------------------------

例如：

condition_norm = condition / pi

范围 [-1,1]

则必须：

condition_rad = condition_norm * pi

condition_rc_rad =
    wrap_phase(condition_rad + u_rad)

condition_rc =
    condition_rc_rad / pi

总之：

禁止直接对归一化 [-1,1] Tensor 使用 2*pi wrapping。

请根据当前项目的真实 normalization 自动适配。

==================================================
5. Re-corrupted branch
==================================================

使用完全相同的：

x_t
timesteps
model parameters

但 condition 换成：

condition_rc

第二次 forward：

epsilon_pred_rc =
    model(x_t, timesteps, condition_rc)

然后同样恢复：

x0_pred_rc =
    predict_x0_from_epsilon(
        x_t,
        epsilon_pred_rc,
        timesteps,
        noise_scheduler,
    )

因此一次 training iteration 中：

clean branch:
    (x_t, t, condition)
        -> epsilon_pred
        -> x0_pred

re-corruption branch:
    (x_t, t, condition_rc)
        -> epsilon_pred_rc
        -> x0_pred_rc

两个 branch 必须使用同一个 model，
不是两个 diffusion network。

==================================================
6. Teacher / Student 关系
==================================================

定义：

Teacher:
    clean condition -> x0_pred

Student:
    re-corrupted condition -> x0_pred_rc

Teacher 使用 stop-gradient：

teacher_x0 = x0_pred.detach()

注意：

只 detach teacher。

绝对不能：

x0_pred_rc.detach()

否则 HRPC 无法训练 student branch。

clean branch 本身仍然通过：

L_eps
L_phase

正常更新梯度。

==================================================
7. HRPC Level 1:
Absolute Phase Consistency
==================================================

定义：

L_abs =
    L1(
        x0_pred_rc,
        teacher_x0
    )

即：

L_abs =
    mean(
        abs(
            x0_pred_rc
            - teacher_x0
        )
    )

作用：

要求 wrapped condition 被重新污染以后，
恢复的 absolute phase 与 clean-condition prediction 保持一致。

==================================================
8. HRPC Level 2:
Gradient Structural Consistency
==================================================

实现一个可微 spatial gradient operator。

推荐 forward difference：

grad_x =
    image[..., :, 1:]
    - image[..., :, :-1]

grad_y =
    image[..., 1:, :]
    - image[..., :-1, :]

或者使用当前项目已有 gradient operator。

不要为了 shape 对齐随意 padding，
clean 和 rc branch 使用完全相同的 gradient 实现即可。

计算：

grad_student =
    gradient(x0_pred_rc)

grad_teacher =
    gradient(teacher_x0)

然后：

L_grad =
    L1(grad_student_x, grad_teacher_x)
    +
    L1(grad_student_y, grad_teacher_y)

建议除以 2：

L_grad =
    0.5 * (
        L1(gx_student, gx_teacher)
        +
        L1(gy_student, gy_teacher)
    )

==================================================
9. HRPC Level 3:
Wrapped / Periodic Consistency
==================================================

这一项必须体现 phase 的 2*pi periodicity。

不要简单计算：

L1(x0_pred_rc, teacher_x0)

因为这已经由 L_abs 完成。

定义 residual：

delta =
    x0_pred_rc - teacher_x0

将 delta 转换到真实 phase radian domain。

然后：

wrapped_delta =
    Wrap(delta)

定义：

L_wrap =
    mean(abs(wrapped_delta))

也可以使用更加平滑的 circular distance：

L_wrap =
    mean(
        1 - cos(delta)
    )

请默认实现两种模式：

wrap_loss_type = "atan2_l1"
或
wrap_loss_type = "cosine"

其中默认：

wrap_loss_type = "atan2_l1"

atan2_l1：

wrapped_delta =
    atan2(
        sin(delta),
        cos(delta)
    )

L_wrap =
    mean(abs(wrapped_delta))

cosine：

L_wrap =
    mean(1 - cos(delta))

注意：

delta 必须是 radian domain。

如果 x0 被归一化，
必须先反映射到 phase radian domain。

==================================================
10. HRPC Loss
==================================================

定义：

L_HRPC =
    L_abs
    + lambda_grad * L_grad
    + lambda_wrap * L_wrap

第一版默认：

lambda_grad = 0.5
lambda_wrap = 0.5

即：

L_HRPC =
    L_abs
    + 0.5 * L_grad
    + 0.5 * L_wrap

==================================================
11. Final Loss
==================================================

最终：

L_total =
    L_eps
    + lambda_phase * L_phase
    + lambda_hrpc * L_HRPC

第一版默认：

lambda_phase = 0.01
lambda_hrpc = 0.05
lambda_grad = 0.5
lambda_wrap = 0.5

因此：

L_total =
    L_eps
    + 0.01 * L_phase
    + 0.05 * (
          L_abs
          + 0.5 * L_grad
          + 0.5 * L_wrap
      )

请把所有 lambda 做成 config / argparse 参数，
不要硬编码到训练循环。

例如：

--lambda_phase 0.01
--lambda_hrpc 0.05
--lambda_grad 0.5
--lambda_wrap 0.5
--rc_sigma ...
--wrap_loss_type atan2_l1

==================================================
12. Re-corruption noise sigma
==================================================

不要直接假设 sigma_rc。

根据当前 condition normalization 实现参数：

rc_sigma

如果 condition 最终在 radian domain 中重新污染：

u_rad =
    torch.randn_like(condition_rad)
    * rc_sigma

第一版可以默认：

rc_sigma = 0.1 rad

但必须允许通过 config 修改。

例如：

--rc_sigma 0.1

后续我要测试：

0.05
0.10
0.20
0.30 rad

因此不要写死。

==================================================
13. 一个非常重要的问题：
不要污染无效区域
==================================================

如果 dataset 中存在：

mask
invalid pixels
NaN
background mask
coherence mask

请检查当前数据处理。

如果 condition 存在无效区域，
不要因为 re-corruption 改变其 mask 语义。

如果已有 valid_mask：

u = u * valid_mask

并在 loss 中根据项目当前逻辑决定是否只统计 valid pixels。

不要自己创造 mask；
只有项目原本存在 mask 时才沿用。

==================================================
14. Loss 模块封装
==================================================

尽量把实现封装到独立文件/函数，例如：

class HRPCLoss(nn.Module):

    def __init__(
        self,
        lambda_phase=0.01,
        lambda_hrpc=0.05,
        lambda_grad=0.5,
        lambda_wrap=0.5,
        rc_sigma=0.1,
        wrap_loss_type="atan2_l1",
    ):
        ...

但如果当前项目已有 loss.py / criterion.py，
优先集成到现有结构。

不要为了这个功能大规模重构项目。

==================================================
15. 建议返回的 Loss 字典
==================================================

返回：

{
    "loss_total": ...,
    "loss_epsilon": ...,
    "loss_phase": ...,
    "loss_hrpc": ...,
    "loss_abs": ...,
    "loss_grad": ...,
    "loss_wrap": ...,
}

训练时真正 backward：

loss_total

==================================================
16. W&B / TensorBoard logging
==================================================

如果项目已有 W&B：

记录：

train/loss_total
train/loss_epsilon
train/loss_phase
train/loss_hrpc
train/loss_abs
train/loss_grad
train/loss_wrap

如果已有 TensorBoard/logger，
沿用现有系统。

不要重复初始化 wandb。

另外建议记录：

train/x0_pred_mean
train/x0_pred_rc_mean
train/rc_noise_std

方便检查 re-corruption 是否正常。

==================================================
17. AMP / Accelerate
==================================================

当前项目如果使用：

accelerate
gradient accumulation
mixed precision
GradScaler
DDP

全部保持原有实现。

不要改成：

loss.backward()

如果当前项目原来使用：

accelerator.backward(loss)

则继续：

accelerator.backward(loss_total)

特别检查第二次 model forward 是否与 gradient accumulation /
DDP / AMP 兼容。

==================================================
18. 计算量说明
==================================================

HRPC 会产生两个 model forward：

Forward A:
    model(x_t, t, condition)

Forward B:
    model(x_t, t, condition_rc)

因此训练计算量大约增加一次 UNet forward。

但是：

不增加第二套模型参数；
不增加第二个 optimizer；
不训练 teacher network；
不使用 EMA teacher（除非项目本来就有 EMA）。

Teacher 只是：

teacher_x0 = x0_pred.detach()

==================================================
19. 必须做的 sanity checks
==================================================

实现以后自动检查：

TEST 1
如果：

condition_rc == condition

理论上：

epsilon_pred_rc ≈ epsilon_pred
x0_pred_rc ≈ x0_pred
L_abs ≈ 0
L_grad ≈ 0
L_wrap ≈ 0

在 model.eval() 且无随机 dropout 的情况下应该成立。

TEST 2
检查：

L_HRPC.backward()

之后 re-corrupted branch 的模型参数存在非零梯度。

TEST 3
确认 teacher：

x0_pred.detach()

没有梯度。

TEST 4
确认 student：

x0_pred_rc

保留 grad_fn。

TEST 5
检查：

loss_total

没有 NaN / Inf。

TEST 6
检查：

alpha_bar_t

shape：

[B,1,1,1]

TEST 7
确认 wrap loss 使用的是 radian phase，
而不是错误地对 [-1,1] normalized phase 使用 2*pi。

TEST 8
rc_sigma=0 时，
HRPC 应非常接近 0
（允许数值误差）。

==================================================
20. Ablation 开关
==================================================

我后面需要做论文消融实验，因此实现开关：

--use_phase_loss
--use_hrpc
--use_abs_consistency
--use_grad_consistency
--use_wrap_consistency

支持以下实验：

A. Baseline
L = L_eps

B. + Phase
L = L_eps + L_phase

C. + RC-Abs
L = L_eps + L_phase + L_abs

D. + RC-Abs-Grad
L = L_eps + L_phase + L_abs + L_grad

E. Full HRPC
L = L_eps + L_phase
    + L_abs + L_grad + L_wrap

但是实际权重仍然按照配置中的 lambda 计算。

关闭 use_hrpc 时：

不要执行第二次 UNet forward，
避免无意义计算开销。

==================================================
21. 保留旧 Loss
==================================================

不要删除现有 loss。

增加配置：

loss_type = "baseline"
loss_type = "phase"
loss_type = "hrpc"

例如：

baseline:
    L_eps

phase:
    L_eps + lambda_phase * L_phase

hrpc:
    完整 HRPC

这样我可以直接使用同一训练代码进行公平消融。

==================================================
22. 代码质量要求
==================================================

要求：

1. 尽量最小修改。
2. 不修改模型结构。
3. 不修改 inference。
4. 不修改 scheduler。
5. 不修改 dataset。
6. 不修改已有 checkpoint 格式，除非 config 新字段需要保存。
7. 保持旧 checkpoint 尽可能兼容。
8. 所有新 Tensor 都放到正确 device。
9. 不产生无意义 CPU-GPU copy。
10. 不使用 numpy 参与 loss 计算。
11. 所有 loss 必须保持 PyTorch autograd。
12. 避免 inplace operation 破坏梯度。

==================================================
23. 最后给我报告
==================================================

修改完成后不要只告诉我“完成”。

请给我：

1. 当前项目原来的 diffusion prediction_type。
2. 当前 x0 / condition 的实际归一化范围。
3. 修改了哪些文件。
4. 修改了哪些函数。
5. 新增了哪些 config 参数。
6. 原来的 loss 数学公式。
7. HRPC 数学公式。
8. epsilon -> x0 的实现代码。
9. wrap_phase 的实现代码。
10. condition re-corruption 的实现代码。
11. teacher/student stop-gradient 的位置。
12. 最终 loss 计算代码。
13. sanity checks 的结果。
14. 是否发现 NaN/Inf 风险。
15. 第二次 forward 是否正确参与梯度传播。
16. 是否保持原有 baseline 模式完全可运行。

如果发现当前项目实现与上述假设冲突，
不要强行修改。
先根据实际代码结构进行适配，并在最终报告中说明。