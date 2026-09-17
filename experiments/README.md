# 3000 ???????????

????? `revision_study.py`??????? `../docs/REVISION_TASKS.zh-CN.md`???????? `../docs/REVISION_DRAFT.zh-CN.md`??????? `results/revision_3000/`?????????

```powershell
.\.venv\Scripts\python.exe -B -u experiments/revision_study.py queue
.\.venv\Scripts\python.exe -B -u experiments/revision_analysis.py watch
```

??????????????????????????????? `last.pth` ????????????????`watch` ???? 25 ???? 41 ???????? GPU ????????? watcher ????????

- `manifest.json` / `data_audit.csv` / `split_integrity.json`?????????????????????????????
- `runs/`???????????????????? 30 ??U3Net 50+20 ??
- `evaluation/`???????? bootstrap ??????????
- `REPORT.zh-CN.md`????????????????????
- `analysis/`????????????????????????????

2026-09-12 ?? InSAR Restormer ?? 6 ?????????WDDM ?????????????? 2 ?????? micro-batch 1??? 6 ????????????? 6?????????????????????????????????????????????????CPU ???????????????????????????????

????????????????????????????

---

# 免训练小样本实验

本目录用于先验证审稿意见中不需要重新训练的实验流程。默认只处理极少量数据，适合当前机器做 smoke test；这些结果只能证明流程可运行，不能直接作为论文中的正式统计结果。

## 已覆盖的检查

- 预训练权重加载与推理
- 原始 MAE / RMSE，以及消除全局整数 `2π` 偏移后的 MAE / RMSE
- 将预测重新缠绕后的圆周误差（cycle consistency）
- 输入与参考绝对相位自身的重缠绕一致性检查
- 多次随机采样所得像素级不确定性及其与误差的 Spearman 相关系数
- 单张图的推理耗时、模型参数量和 CUDA 峰值显存
- 高梯度区域剖面图
- 当前噪声实现与零均值高斯相位噪声的对照审计

## 运行命令

### 传统相位解缠基线（无需训练或权重）

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_traditional_baselines.py `
  --data-root data\SyntheticPUMat128Big `
  --input-kind synthetic `
  --max-samples 5 `
  --warmup 1 `
  --repeats 3 `
  --output experiments\results\traditional_synthetic_smoke
```

噪声实验使用与 `SyntheticPUMatNoise` 相同的零均值高斯相位噪声公式，例如：

```powershell
.\.venv\Scripts\python.exe experiments\evaluate_traditional_baselines.py `
  --data-root data\SyntheticPUMat128Big `
  --input-kind synthetic `
  --noise-snr-db 10 `
  --seed 42 `
  --max-samples 5 `
  --output experiments\results\traditional_synthetic_10db_smoke
```

脚本统一评估二维顺序 Itoh、质量引导最大生成树和最小二乘 Poisson
三种无需训练的传统方法，输出 `report.json`、`per_sample.csv` 和首个样本的
`first_sample.png`。所有精度指标同时保留原始结果和仅消除全局整数 `2π`
偏移后的结果。

SNAPHU/MCF 需要复干涉图、相干系数和等效视数等统计输入。当前数据调用只提供
缠绕相位，因此脚本不会用全 1 相干系数伪造 SNAPHU 结果；拿到相干图后应再作为
独立基线接入。

在项目根目录执行：

```powershell
# 合成数据：1 张图、3 次随机采样
.\.venv\Scripts\python.exe experiments\evaluate_pretrained_smoke.py `
  --config configs\fdu_synpu_128_big.yaml `
  --checkpoint assets\SyntheticPUMat128Big\FduDDPMDiffusion\ckpt\epoch_94.pth `
  --data-root data\SyntheticPUMat128Big `
  --input-kind synthetic `
  --max-samples 1 `
  --draws 3 `
  --output experiments\results\synthetic_epoch94_smoke

# 真实 InSAR 数据：1 张图、3 次随机采样
.\.venv\Scripts\python.exe experiments\evaluate_pretrained_smoke.py `
  --config configs\fdu_dlpu_256_big.yaml `
  --checkpoint assets\InSARDLPUMat256Big\FduDDPMDiffusion\ckpt\epoch_160.pth `
  --data-root data\InSARDLPUMat256Big `
  --input-kind insar-real `
  --max-samples 1 `
  --draws 3 `
  --output experiments\results\insar_real_epoch160_smoke

# 噪声公式审计
.\.venv\Scripts\python.exe experiments\audit_noise_model.py `
  --input data\SyntheticPUMat128Big\test_in\000001.mat `
  --snr-db 10 `
  --seed 42 `
  --output experiments\results\noise_model_smoke
```

输出目录必须不存在或为空，以避免误覆盖历史结果。每次运行都会生成供人阅读的
`summary.md` 和供程序处理的 `report.json`；传统方法额外输出 `per_sample.csv`，
预训练评估还会生成 `maps.png`、`high_gradient_profile.png` 和保存首个样本张量的
`first_sample.pt`。

## 当前 smoke test 结果

以下数字均只来自 1 个样本，不代表最终性能：

| 检查 | 合成数据 epoch 94 | 真实数据 epoch 160 |
|---|---:|---:|
| 参数量 | 13,908,097 | 3,488,833 |
| 5 步推理平均耗时 | 172.52 ms | 189.56 ms |
| CUDA 峰值显存 | 153.25 MB | 219.02 MB |
| 原始 MAE | 0.0452 | 8.4078 |
| 全局 `2π` 对齐后 MAE | 0.0452 | 5.3063 |
| 预测重缠绕圆周 MAE | 0.0451 | 1.4841 |
| 参考相位自身重缠绕圆周 MAE | 约 0 | 2.2331 |
| 平均预测标准差 | 0.00034 | 0.57869 |

真实样本的参考绝对相位重缠绕后，与输入缠绕相位仍有很大差异；估计的圆周常量偏移约为 `3.1166 rad`，校正后圆周 MAE 仍为 `0.9095 rad`。应先查清数据配对、相位符号、常量基准或预处理约定，再使用真实集误差评价模型。

噪声审计中，请求 `10 dB` 时，当前公式的噪声均值为 `-3.1447 rad`，实测相位 SNR 为 `-5.13 dB`；零均值修正版的实测 SNR 为 `9.97 dB`。这表明现有公式含明显的 `-π` 偏置，正式实验前应修正并补做噪声消融。

## 换到高性能机器后的建议

先保持脚本与权重不变，将 `--max-samples` 增加到至少 100，将 `--draws` 增加到 20～50。不同模型和基线必须使用同一份样本清单、同一套对齐规则和同一计时方式；确认真实数据约定后再生成论文表格。
