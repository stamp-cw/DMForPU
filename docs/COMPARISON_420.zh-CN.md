# 同批数据验证：FDU、HF、DCC/WWFCA、DLPU 与 Chen 改进

## 任务与固定协议

用户要求使用上一轮同一批数据比较。数据清单逐字节复制 `experiments/results/chen_hf_diffusion/manifest.json`，不重新抽样：Synthetic 原始 128×128，共 300 训练、60 验证、60 测试。用户所写 ddc 按项目组件名称 DCC 执行。

统一 20 轮、batch 8、AdamW lr=0.0002、weight_decay=0、梯度裁剪 1、FP16；seed=42/43。全部方法通过同一 `chen_hf_study.train` 实现生成固定顺序、观察噪声、扩散噪声和时间步。主监督项统一为归一化绝对相位 MSE；Chen 的 SR/SD 按原改进实验保留。**这是共同筛选训练配方，非各方法原论文最佳/原始 L1 配方**。

扩散统一 DDPM、1000 时间步、sample prediction、5 步单次采样、不裁剪样本。DLPU 为直接回归，只进行一次前向。测试使用无附加噪声、30/20/10/5/0 dB 六档相位噪声；σ²=10^0.1/10^(SNR/10)。使用同一组图、输入扰动与采样种子；图级 MAE/RMSE/NRMSE/PGE/圆周一致性、原始与整数周期对齐指标同源。

**主表统一使用最后第 20 轮权重**，避免 SD 组仅从最后 6 轮选模而其他组从全部训练选模的窗口差异。另输出 best 验证最佳附表；不以测试结果选 checkpoint。

## 8 组模型

| 名称 | 骨干与组件 | 权重来源 |
|---|---|---|
| hf_small | 原生 HF，三层 32/64/64，w/π 条件 | 上轮同数据协议的 hf_base，重新推理 |
| hf_matched | 原生 HF，四层 128/128/128/128，每块 2 层，cross_dim=384；无 DCC/WWFCA | 从头新训 |
| dcc_only | 同规格 HF + sin/cos 双通道条件 | 从头新训 |
| wwfca_only | 项目 FDUNet，单通道 w/π 条件 | 从头新训 |
| fdu | 项目 FDUNet，DCC+WWFCA | 从头新训 |
| dlpu | 项目完整 DLPUNet | 从头新训 |
| chen_full | 上轮小 HF + 物理迭代/CAM/E/SR/SD | 同数据协议的 full，重新推理 |
| chen_no_sparse | 上轮改进中移除稀疏 E | 同数据协议的 no_sparse，重新推理 |

四个同规格扩散组构成 DCC×WWFCA 2×2 消融。额外保留小 HF，使 Chen 改进与其真正基线对照；不能把骨干尺寸差异解释成组件收益。输出实际参数量。

复用只限上轮已经从头训练的同批 420 对数据检查点；加载时严格核对 manifest 哈希、轮数、batch、lr、seed、来源变体以及 state_dict。没有使用 3000 对实验的权重。两种子共新增 10 组训练、复用 6 组训练；final/best 各评估一遍，总计 32 组评估。

## 可重复入口

```powershell
.\.venv\Scripts\python.exe -B -u experiments/compare_420.py queue
.\.venv\Scripts\python.exe -B experiments/compare_420.py report
```

输出目录 `experiments/results/comparison_420/`，含任务队列、检查点、逐图指标、数据来源和报告。现有训练 checkpoint 支持中断恢复；不要重复启动多个队列。

## 已知解释边界

- 固定种子和噪声不保证 CuDNN/FP16 跨进程逐位一致，两个种子不能代替稳定性验证。
- 小数据、20 轮的排名不代表充分收敛或全量泛化，原生大骨干可能更难在该预算下训练。
- Chen 模型获得合成噪声 σ，且 SR/SD 增加前向次数。组件与信息输入、计算成本均应披露，不能声称等计算量、等条件信息。
- 上轮 no_sparse 是探索性候选，并非独立新测试集确认的最佳方法。

状态：2026-09-12 14:55 完成。新增 10 组训练均实际运行 20 轮，32/32 组 final/best 评估完成；训练损失有限、每轮均有成功更新、manifest 哈希一致、60 张测试和六档噪声齐全。8 项相关测试通过。核验文件为 `experiments/results/comparison_420/completion_audit.json`。

## 完成后的主要观察

- final 主表中 DLPU 的两种子平均 clean/10 dB/0 dB MAE 分别为 2.5015/2.5034/2.5314，最低且噪声变化小；但 clean PGE=0.3045，在主表方法中最高。
- 同规格 HF 为 3.6069/3.7093/4.2133；DCC-only 为 2.7295/2.8574/4.7521。DCC 改善低噪声精度和 PGE，却在两个种子的 0 dB 上都退化。
- WWFCA-only 为 4.6880±2.5520，种子方差很大。完整 FDU 为 3.2033±0.5425，优于同规格 HF 的平均 clean MAE，但差于 DCC-only；说明 DCC 与 WWFCA 有明显交互，不能把两者收益简单相加。
- Chen no_sparse 为 3.2057/3.2988/3.8648，与 FDU 的 clean 均值接近、0 dB 更低；Chen full 为 3.7103/3.7372/4.5222。本批次仍未支持“所有 Chen 组件叠加最好”。
- 统一 final 是主结论。best 表中 DLPU clean MAE=1.9218，FDU=2.7599，但选中轮次不同，只作训练稳定性参考。

完整数值、逐种子 2×2 差值和图见 `experiments/results/comparison_420/REPORT.zh-CN.md` 与 `figures/`。
