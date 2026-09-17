# Chen 2024 启发的原始 HF 扩散改进

## 来源与边界

- 本地论文：`refence_paper/Chen 等 - 2024 - Unsupervised Deep Unrolling Networks for Phase Unwrapping.pdf`，已核对第 4–5 页式 (8)–(14)。
- 官方仓库：[Unsupervised-PU](https://github.com/chenzhile1999/Unsupervised-PU)，本地审计快照 `47c0af31fee5ee7f442aa433a5fb3483a4376334`，重点核对 `modules/network.py`、`data.py`、`metrics.py`、`train.py`。
- 基底：项目 `diffusion/old/ddpm_diffusion.py` 引用的 HF `UNet2DConditionModel` 和 `DDPMScheduler`。新代码直接实例化 HF 类；没有导入 FDU、FDUNet、WWFCA、小波模块或任何旧权重。
- 新文件 `diffusion/chen_hf_diffusion.py` 提供独立的 `ChenHFDiffusion(nn.Module)`；独立入口为 `experiments/chen_hf_study.py`，不复用旧 FDU 训练包装器。

## 迁移的机制

1. **梯度域物理迭代及稀疏异常值**：参考式 (8)，拟合 `Gφ − wrap(Gw) + E`。HF 给出的绝对相位估计作为先验，经过 3 个阶段、每阶段 3 次加速梯度迭代。E 由梯度残差、小卷积网络和软阈值产生；梯度与伴随采用不跨边界的有限差分。修正保持空间均值，无法凭空决定绝对常数偏移。
2. **噪声条件自适应**：CAM 根据 `log(1+σ)` 产生有界步长、先验混合权重和稀疏阈值。无 CAM 消融仍保留可学习的逐阶段常量，检验的是是否依赖噪声条件，而不是是否存在参数。
3. **配对再加噪的梯度自重建**：参考式 (13) 和官方 `data.py`，先构造 `w_plus=wrap(w+σz)`、有效 `u=w_plus−w`，监督梯度目标为 `G(w−u)`，损失为圆周残差平方。多阶段按后段较高权重平均。
4. **冻结教师梯度蒸馏**：从本轮同一模型第 14 轮后的状态复制教师，最后 6 轮增加蒸馏。学生用原观测，教师用再加噪观测，共享同一扩散时间和扩散状态，避免把扩散随机性混入比较。

## 必须披露的适配差异

- 论文不是扩散模型。这里以 HF 的 x₀ 估计充当相位先验，没有复用原 U3Net 的 SubNN_X；物理迭代属于新适配，不能直接继承原论文的精度结论。
- 所有变体始终保留配对数据的监督 x₀ MSE；SR/SD 是附加项，权重均为 0.05。训练目标依赖真值，**不是无监督方法**。式 (12) 的无偏性结论也不能原封不动套用，因为扩散状态含真值信息。
- 论文式 (14) 写梯度平方 L2、联合损失；官方 `metrics.py` 实际用梯度 L1，`train.py` 采用先 SR、后冻结教师 SD 的两阶段。这里采用代码中的梯度 L1、固定教师，但保留监督主损失和 SR；这些改动明示，不声称完全复现。
- HF 骨干宽度为 32/64/64、每块 1 层，作为有限算力小批筛选；所有变体骨干相同，共用相同初始化种子。输入只有扩散状态和 `w/π`，不使用 FDU 的双通道 sin/cos 条件。
- 标准 DDPM，1000 训练时间步、直接 x₀ prediction；测试 5 个反向步、1 次采样。调度器不裁剪样本；相位缩放固定为 `[0,14π]`，不根据测试真值拟合。

## 固定小批协议

从已有固定清单取 Synthetic 域前 300 对训练、60 对验证、60 对官方测试，共 420 对、原始 128×128；三者路径不交叉。此轮先检验机制，不把它冒充 InSAR 或真实数据结果。

七个变体 × seed 42/43，每个从头训练 20 轮、batch 8、AdamW lr=0.0002、weight_decay=0、梯度裁剪 1、FP16：

| 变体 | 物理迭代 | SR | SD | 稀疏 E | 噪声 CAM |
|---|---|---|---|---|---|
| hf_base | 无 | 无 | 无 | — | — |
| physics | 有 | 无 | 无 | 有 | 有 |
| sr | 无 | 有 | 无 | — | — |
| physics_sr | 有 | 有 | 无 | 有 | 有 |
| full | 有 | 有 | 有 | 有 | 有 |
| no_sparse | 有 | 有 | 有 | 无 | 有 |
| no_cam | 有 | 有 | 有 | 有 | 无（可学习常量） |

每轮所有变体使用相同的顺序、观测噪声、扩散噪声、时间步和再加噪随机数。50% 不额外加噪，其余均分 30/20/10/5/0 dB。σ²=10^0.1/10^(SNR/10)，固定参考功率；clean 的 σ=0。验证选模使用 clean 与 10 dB 的平均整数周期对齐 MAE；SD 组只从蒸馏阶段选模。测试集不选超参数。

评估六档噪声，输出原始/整数周期对齐 MAE、RMSE、NRMSE、PGE、圆周一致性及图级 bootstrap CI。所有随机种子分别保留；不做逐图幅值拟合。各组额外前向次数不同，应结合训练耗时理解，而非宣称等计算量。

## 运行与输出

```powershell
.\.venv\Scripts\python.exe -B -u experiments/chen_hf_study.py queue
```

输出 `experiments/results/chen_hf_diffusion/`：`manifest.json`、`runs/`、`evaluation/`、`status.json`、`REPORT.zh-CN.md`。检查点只在本轮目录内恢复；完整结果以实际文件为准。旧 `revision_3000` 与本轮骨干、样本量和预算不同，不直接用其分数做公平排名。

状态：实现和 5 项机制测试通过，正在安排独立 GPU 训练；尚未证实提升精度。

## ???????

?????????????? GPU FP16 / CuDNN benchmark ???????????????? seed42 ? physics_sr ? full ???? 14 ?????????? 0.00134??? MAE ???? 0.2191??????? SD????????? SD??????????????????????????????????????????????????? 14 ????????????????????

## ??????2026-09-12?

14/14 ???????? 10:07 ????????? 20 ??6 ? SD ????????? 6 ??????????????????????????????????? completion_audit.json?

????? clean MAE?HF ?? 3.6809?physics 3.5296?physics_sr 3.5039?full 3.5532?no_sparse 3.2057?no_sparse ?????????? 12.9%?0 dB ??? 10.1%?? seed42 ?????seed43 ???????????????????????full ??? PGE=0.1605????? 0.2255????? MAE ???

?????????????????????????? `experiments/results/chen_hf_diffusion/REPORT.zh-CN.md` ? `figures/`?????? 3000 ????? 12:39 ???????12:55 ???????
