请修改我当前的 diffusion phase unwrapping 训练代码，实现论文
“Phase unwrapping and pixel-wise uncertainty evaluation based on a conditional diffusion model”
中的联合损失函数。

不要修改现有网络结构、数据加载方式、noise scheduler 和推理流程，
只修改训练阶段的 loss 计算，并尽量保持现有代码结构。

==============================
1. 当前训练目标
==============================

模型采用 epsilon-prediction：

epsilon_pred = model(x_t, t, condition)

其中：
- x0：真实 unwrapped phase，已经归一化到模型训练范围
- epsilon：随机采样的真实 Gaussian noise
- x_t：由 x0 在 timestep t 加噪得到
- epsilon_pred：网络预测的噪声
- alpha_bar_t：diffusion scheduler 对应 timestep t 的 cumulative alpha

标准扩散噪声损失为：

L_eps = MSE(epsilon_pred, epsilon)

==============================
2. 从 epsilon_pred 重建 x0
==============================

根据：

x_t = sqrt(alpha_bar_t) * x0
    + sqrt(1 - alpha_bar_t) * epsilon

利用网络预测的 epsilon_pred 反推出：

x0_pred =
    (x_t - sqrt(1 - alpha_bar_t) * epsilon_pred)
    / sqrt(alpha_bar_t)

注意：

1. alpha_bar_t 必须从当前 scheduler 获取，
   不允许自己重新计算一套 diffusion schedule。

2. alpha_bar_t 要根据 batch 中每个样本的 timestep t 正确索引。

3. reshape/broadcast 成：
   [B, 1, 1, 1]
   以适配 [B,C,H,W] 图像 Tensor。

4. alpha_bar_t、x_t、epsilon_pred 必须：
   - device 一致
   - dtype 一致

5. 如果当前代码使用 AMP/fp16，需要避免 sqrt 和除法导致数值不稳定。
   必要时使用 clamp：
   alpha_bar_t = alpha_bar_t.clamp(min=1e-8)

==============================
3. x0 reconstruction loss
==============================

计算：

L_x0 = L1(x0_pred, x0)

即：

L_x0 = mean(abs(x0_pred - x0))

这里比较的是归一化后的 unwrapped phase。

不要在 loss 内部进行反归一化。

==============================
4. 最终 Loss
==============================

按照论文设置：

lambda_x0 = 0.01

最终：

L_total = L_eps + lambda_x0 * L_x0

即：

L_total =
    MSE(epsilon_pred, epsilon)
    + 0.01 * L1(x0_pred, x0)

使用 L_total 进行：

optimizer.zero_grad()
L_total.backward()
optimizer.step()

如果当前代码使用：
- accelerator.backward()
- GradScaler
- gradient accumulation

则保持当前训练框架，不要破坏原来的反向传播逻辑。

==============================
5. 训练日志
==============================

不要只记录 total loss。

分别记录：

loss_total
loss_epsilon
loss_x0

例如：

loss_epsilon = ...
loss_x0 = ...
loss_total = loss_epsilon + 0.01 * loss_x0

如果使用 wandb，则分别记录：

wandb.log({
    "train/loss_total": loss_total.item(),
    "train/loss_epsilon": loss_epsilon.item(),
    "train/loss_x0": loss_x0.item(),
})

如果项目已有 logging 机制，则接入现有 logging，
不要额外重复初始化 wandb。

==============================
6. 实现方式
==============================

优先封装成独立函数，例如：

def diffusion_phase_loss(
    epsilon_pred,
    epsilon,
    x_t,
    x0,
    timesteps,
    noise_scheduler,
    lambda_x0=0.01,
):
    ...

返回：

{
    "loss": loss_total,
    "loss_epsilon": loss_epsilon,
    "loss_x0": loss_x0,
    "x0_pred": x0_pred,
}

或者根据当前项目代码风格采用 tuple/dataclass。

不要为了实现这个 loss 大规模重构项目。

==============================
7. Diffusers scheduler兼容
==============================

如果项目使用 HuggingFace diffusers，例如：

DDPMScheduler

优先直接使用：

noise_scheduler.alphas_cumprod

根据 timesteps 获取：

alpha_bar_t = noise_scheduler.alphas_cumprod[timesteps]

然后 reshape：

alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)

计算：

sqrt_alpha_bar = alpha_bar_t.sqrt()
sqrt_one_minus_alpha_bar = (1.0 - alpha_bar_t).clamp(min=0).sqrt()

x0_pred = (
    x_t
    - sqrt_one_minus_alpha_bar * epsilon_pred
) / sqrt_alpha_bar.clamp(min=1e-8)

==============================
8. 重要检查
==============================

实现后请主动检查以下问题：

A. 当前模型到底是：
prediction_type="epsilon"
还是：
prediction_type="sample"
或：
prediction_type="v_prediction"

只有 epsilon prediction 才直接使用上述公式。

如果不是 epsilon prediction，不要强行修改，
先指出当前 prediction_type。

B. 检查当前代码中的 x0 是否确实是 ground-truth unwrapped phase。

C. 检查 x0 和 x0_pred 是否处于相同归一化尺度。

D. 检查 timestep indexing 是否正确。

E. 检查 alpha_bar_t 的 shape 是否能正确 broadcast。

F. 检查 AMP 下是否出现 NaN/Inf。

G. 不要对 x0_pred 使用 detach()，
因为 L_x0 必须能够通过 x0_pred 反向传播到 epsilon_pred。

H. 不要对 epsilon_pred 做 detach()。

==============================
9. 最后输出
==============================

修改完成后，请告诉我：

1. 修改了哪些文件；
2. 修改了哪些函数；
3. 原来的 loss 是什么；
4. 新的 loss 是什么；
5. 给出最终数学公式；
6. 给出核心修改代码；
7. 检查 lambda_x0=0.01 是否正确生效；
8. 检查 L_x0 是否确实参与梯度反向传播；
9. 不要修改与 loss 无关的模型结构和训练参数。