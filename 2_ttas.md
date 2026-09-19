请在当前相位解缠项目的 metrics/evaluation 模块中实现 3 种 AU（Accuracy of Unwrapping）评价指标：

1. Raw AU
2. Integer-Aligned AU（I-AU）
3. Range-Aligned AU（RA-AU）

要求三种指标使用完全相同的 AU 判定规则，仅对预测相位 pred 的“对齐方式”不同。

==================================================
一、符号定义
==================================================

对于单幅图像：

pred = \hat{\psi}：模型预测的解缠相位
gt   = \psi：Ground Truth 解缠相位

图像尺寸为 H × W。

AU 的容差系数：

eta = 0.05

默认使用 5% 阈值。

定义 GT 最小值：

psi_min = min(gt)

对于经过某种方式对齐后的预测结果 pred_align，定义逐像素绝对误差：

E(x,y) = |pred_align(x,y) - gt(x,y)|

定义逐像素允许误差：

T(x,y) = |gt(x,y) - psi_min| * eta

Binary Error Map（BEM）定义为：

BEM(x,y) =
    1,  if E(x,y) <= T(x,y)
    0,  otherwise

最终：

AU = sum(BEM) / (H * W) * 100%

AU 越高越好。

注意：
三种 AU 的区别仅仅是 pred_align 的计算方法不同。

==================================================
二、Raw AU
==================================================

Raw AU 不进行任何相位对齐：

pred_align = pred

因此：

E_raw(x,y) = |pred(x,y) - gt(x,y)|

BEM_raw(x,y) =
    1, if |pred(x,y)-gt(x,y)|
          <= |gt(x,y)-min(gt)| * eta
    0, otherwise

Raw_AU = mean(BEM_raw) * 100%

该指标评价模型对“绝对相位值”的直接恢复精度。

实现函数：

raw_au(pred, gt, eta=0.05)

==================================================
三、Integer-Aligned AU（I-AU）
==================================================

I-AU 用于消除相位解缠固有的全局 2πk 整数周期偏移。

只能进行：

..., -4π, -2π, 0, +2π, +4π, ...

这样的全局对齐，不允许任意实数平移，也不允许缩放。

首先计算：

d(x,y) = pred(x,y) - gt(x,y)

估计全局整数周期：

k* = round(
    median(pred - gt) / (2*pi)
)

然后：

pred_IA = pred - 2*pi*k*

注意：
k* 必须针对“每幅图像独立计算”，不能对整个 batch 计算一个共同的 k。

然后计算：

E_IA(x,y) = |pred_IA(x,y) - gt(x,y)|

BEM_IA(x,y) =
    1, if E_IA(x,y)
          <= |gt(x,y)-min(gt)| * eta
    0, otherwise

最终：

I_AU = mean(BEM_IA) * 100%

实现函数：

integer_aligned_au(pred, gt, eta=0.05)

建议同时返回：

{
    "au": I_AU,
    "aligned_pred": pred_IA,
    "k": k_star
}

该指标主要评价：
消除 global 2πk ambiguity 后的真实 phase-unwrapping correctness。

==================================================
四、Range-Aligned AU（RA-AU）
==================================================

RA-AU 使用 U3Net 风格的 Min-Max Range Alignment。

它不仅消除全局平移误差，同时消除预测相位与 GT 之间的整体幅值尺度差异。

对于每幅图像分别计算：

pred_min = min(pred)
pred_max = max(pred)

gt_min = min(gt)
gt_max = max(gt)

首先将 pred 归一化到 [0,1]：

pred_norm =
    (pred - pred_min)
    /
    (pred_max - pred_min + eps)

然后映射到 GT 的数值范围：

pred_RA =
    pred_norm * (gt_max - gt_min)
    + gt_min

即：

pred_RA =
    (pred - pred_min)
    / (pred_max - pred_min + eps)
    * (gt_max - gt_min)
    + gt_min

其中：

eps = 1e-8

用于防止 pred_max == pred_min 时除零。

然后：

E_RA(x,y) = |pred_RA(x,y) - gt(x,y)|

BEM_RA(x,y) =
    1, if E_RA(x,y)
          <= |gt(x,y)-gt_min| * eta
    0, otherwise

最终：

RA_AU = mean(BEM_RA) * 100%

实现函数：

range_aligned_au(pred, gt, eta=0.05, eps=1e-8)

建议同时返回：

{
    "au": RA_AU,
    "aligned_pred": pred_RA
}

RA-AU 主要评价：
去除 global offset 和 global scale discrepancy 后，
预测相位场的空间结构/相对形态恢复能力。

注意：
RA-AU 是基于原 AU 定义扩展出的 Range-Aligned AU，
不要在代码或论文中将其描述为 U3Net 官方 AU。
U3Net 风格仅指这里采用的 Min-Max range alignment。

==================================================
五、三种指标的统一关系
==================================================

三种 AU 必须共用同一个 AU 核心计算函数：

AU(pred_align, gt, eta)

区别仅为：

Raw AU:
    pred_align = pred

I-AU:
    pred_align = pred - 2*pi*k*

RA-AU:
    pred_align =
        (pred-pred_min)
        /(pred_max-pred_min+eps)
        *(gt_max-gt_min)
        +gt_min

因此推荐代码结构：

_compute_au(aligned_pred, gt, eta)

raw_au(...)
integer_aligned_au(...)
range_aligned_au(...)

不要复制三套 BEM/AU 代码。

==================================================
六、Batch 处理要求
==================================================

输入需要支持：

[B,H,W]
[B,1,H,W]

如果是 [B,1,H,W]，可以内部 squeeze channel。

非常重要：

所有 alignment 参数必须 PER-IMAGE 计算。

也就是每张图分别计算：

k*
pred_min
pred_max
gt_min
gt_max

禁止在整个 batch 上计算全局 min/max/median。

最终 batch 指标：

AU_batch = mean(AU_i)

同时最好返回每幅图的 AU：

per_image_au = [AU_1, AU_2, ..., AU_B]

==================================================
七、最终评估输出
==================================================

evaluation 时同时报告：

Raw AU (%)
Integer-Aligned AU / I-AU (%)
Range-Aligned AU / RA-AU (%)

例如：

Raw AU          : 92.31 %
Integer-Aligned AU : 97.84 %
Range-Aligned AU   : 98.52 %

建议保存到 CSV：

model,
raw_au,
integer_aligned_au,
range_aligned_au

==================================================
八、测试
==================================================

请增加 unit tests。

Case 1：
pred = gt

预期：

Raw AU = 100%
I-AU = 100%
RA-AU ≈ 100%

Case 2：

pred = gt + 2*pi

预期：

I-AU = 100%

Raw AU 应明显降低。

Case 3：

pred = gt + 1.0

预期：

I-AU 不应该自动消除这个 +1.0 rad 偏移。

用于验证 I-AU 只允许 2πk alignment。

Case 4：

pred = 0.5 * gt + 10

预期：

RA-AU ≈ 100%

而 Raw AU 和 I-AU 不应达到 100%。

用于验证 RA-AU 能够消除全局 affine range discrepancy。

Case 5：

构造局部区域：

pred = gt

但右半区域：

pred[:, W//2:] += 2*pi

此时 I-AU 和 RA-AU 都不应该达到 100%。

因为这是局部 phase-unwrapping/cycle-slip error，
不能被 global alignment 消除。

==================================================
九、命名
==================================================

代码和结果统一使用：

raw_au
integer_aligned_au
range_aligned_au

论文显示名称：

Raw AU
Integer-Aligned AU (I-AU)
Range-Aligned AU (RA-AU)

不要把 RA-AU 写成 U3Net AU。