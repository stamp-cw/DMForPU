# 相位解缠统一评测指标

## U3Net 式对齐

对每一张预测相位 `pred` 单独执行 min–max 仿射对齐，使其动态范围与对应的真实无噪声相位 `target` 一致：

\[
pred_{u3}=\frac{pred-\min(pred)}{\max(pred)-\min(pred)}
(\max(target)-\min(target))+\min(target).
\]

分母使用 `1e-12` 下限，以保证常数预测不会产生 NaN。所有方法，包括深度模型、扩散模型和传统方法，使用同一个实现。

对齐后记录：

- `u3_aligned_mae`：`mean(abs(pred_u3 - target))`，单位 rad。
- `u3_aligned_rmse`：`sqrt(mean((pred_u3 - target)^2))`，单位 rad。
- `u3_aligned_nrmse`：`u3_aligned_rmse / (max(target)-min(target))`，无量纲。乘以 100 后为百分数。
- `u3_aligned_ssim`：在 `pred_u3` 与 `target` 之间计算的结构相似性，越接近 1 越好。

SSIM 使用 11×11 高斯窗口、标准差 1.5、`K1=0.01`、`K2=0.03`，每张图的数据范围采用真实相位的 `max-min`。边界 5 个像素不纳入局部 SSIM 平均。

## 与其他对齐的区别

- `raw_*` 不进行对齐，保留绝对相位偏移。
- `integer_aligned_*` 只消除全局整数个 `2π` 偏移。
- `mean_aligned_*` 只消除全局平均偏移。
- `u3_aligned_*` 同时消除加性偏移和正比例尺度误差，用于对应 U3Net 的评测方式。

最佳权重仍按验证集 `mean_aligned_mae` 选择，避免改变既定模型选择规则。`u3_aligned_*` 作为测试和对比报告中的补充指标。

统一实现位于 `utils/phase_metrics.py`。

## AU 指标

三种 Accuracy of Unwrapping 指标共用同一个逐像素判定规则，默认 `eta=0.05`：

\[
T(x,y)=|target(x,y)-\min(target)|\eta,
\qquad
AU=\operatorname{mean}(|pred_{align}-target|\leq T)\times100\%.
\]

所有最小值、中位数和对齐参数均逐图计算，最终结果为每张图 AU 的平均值。

- `raw_au`：不对预测进行任何对齐，用于衡量绝对相位恢复能力。
- `integer_aligned_au`：逐图计算 `k=round(median(pred-target)/(2π))`，再使用 `pred-2πk`，只消除全局整数周偏移。
- `range_aligned_au`：使用与 U3Net NRMSE 相同风格的逐图 min–max 范围对齐后计算 AU，同时消除全局偏移和正比例尺度差异。

三个 AU 均以百分数保存，越高越好。`range_aligned_au` 是本项目基于原 AU 定义扩展的 RA-AU，不能称为 U3Net 官方 AU。
