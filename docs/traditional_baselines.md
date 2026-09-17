# 传统相位解缠基线实验

本项目提供三种不需要训练或预训练权重的传统二维相位解缠基线：

| 命令名 | 方法 | 用途与限制 |
|---|---|---|
| `itoh` | 行、列顺序 Itoh 路径积分 | 最简单、速度快；遇到残差时容易沿扫描方向传播错误 |
| `quality_guided_mst` | 基于二阶相位差可靠度的最大生成树积分 | 优先连接可靠像素，避免固定扫描顺序；当前实现不使用相干图 |
| `least_squares` | DCT/Poisson 无权最小二乘 | 全局优化缠绕梯度；在含残差数据上不保证逐像素重缠绕完全等于输入 |

代码位于 `traditional/phase_unwrapping.py`，统一评测入口是
`experiments/evaluate_traditional_baselines.py`。算法仅接收缠绕相位，不读取真实标签；
真实标签只用于结束后的指标计算和全局整数 `2π` 偏移对齐。

## 报告指标

- MAE、RMSE；
- 全局整数 `2π` 对齐后的 MAE、RMSE、NRMSE；
- 相位梯度误差 PGE；
- 重缠绕圆周 MAE、RMSE，以及超过 `π/10`、`π/4` 的像素比例；
- 单张 CPU 推理时间。

论文主表应优先报告全局 `2π` 对齐后的指标，因为绝对相位解允许相差一个全局整数
周期；同时保留原始指标和估计的整数偏移，避免对齐掩盖参考基准问题。

## 论文实验建议

正式实验应在完全相同的样本列表上比较：本文方法、DLPU、U3Net、Restormer、
Uformer、SqdLstm 和上述传统方法。无噪、0、5、10、20、30 dB 分别统计完整测试集
均值和标准差。传统方法运行在 CPU，因此延迟表需要明确硬件和线程数，不应与 GPU
延迟混为同一列而不加说明。

当前 smoke test 表明，无噪的平滑 SyntheticPU 样本经全局 `2π` 对齐后可被三种传统
方法近乎精确恢复。这意味着论文不能只依靠这类容易样本论证深度模型优势；应重点展示
高噪声、含残差、高梯度和真实 TanDEM-X 场景。

## SNAPHU / MCF 边界

SNAPHU 是统计代价网络流方法，除了复干涉图之外还需要相干系数、等效视数以及适当的
统计代价模式。当前 MAT 调用只提供缠绕相位和参考绝对相位，因此未生成伪造的
SNAPHU/MCF 数字。获得原始复干涉图和相干图后，再把 SNAPHU 的 `mcf` 与 `mst`
初始化模式作为独立实验接入，并记录全部配置。

## 建议引用

1. K. Itoh, “Analysis of the phase unwrapping algorithm,” *Applied Optics*, 21,
   2470 (1982). DOI: 10.1364/AO.21.002470.
2. M. A. Herráez, D. R. Burton, M. J. Lalor, and M. A. Gdeisat, “Fast
   two-dimensional phase-unwrapping algorithm based on sorting by reliability
   following a noncontinuous path,” *Applied Optics*, 41, 7437–7444 (2002).
   DOI: 10.1364/AO.41.007437.
3. D. C. Ghiglia and L. A. Romero, “Robust two-dimensional weighted and
   unweighted phase unwrapping that uses fast transforms and iterative methods,”
   *JOSA A*, 11, 107–117 (1994). DOI: 10.1364/JOSAA.11.000107.
