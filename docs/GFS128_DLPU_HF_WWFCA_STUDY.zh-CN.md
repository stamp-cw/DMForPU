# GFS128：DLPU、HF diffusion 与 WWFCA-only 实验

## 实验范围

本轮只训练以下三种方法，不训练 DCC、FDU 或 Chen 系列：

| 展示名称 | 代码方法 | 轮数 | batch | 说明 |
|---|---|---:|---:|---|
| DLPU | `dlpu` | 100 | 32 | 直接导入原仓库 `Phase_unwrapping_by_U-Net` 的网络 |
| HF diffusion | `hf_matched` | 300 | 8 | 四层 128 通道 HF `UNet2DModel`，无 DCC、无注意力 |
| WWFCA-only diffusion | `wwfca_only` | 300 | 8 | 相同宽度和层数，加入修正后的 WWFCA，无 DCC |

使用 `hf_matched` 作为 HF 基线，是为了让 HF 与 WWFCA-only 的深度、宽度、输入条件和扩散训练配置一致。两者的实验变量是 WWFCA 频域交叉注意力。

## 数据与选择规则

- 数据集：`data/GFS128/train.h5`，共 5000 张，包含 clean、0、5、10、20、30 dB。
- 固定随机种子 42，将训练文件固定划分为 4500 张训练和 500 张验证。
- 测试集：匹配场景的 clean、0、5、10、20、30 dB，每种 1000 张。
- 每轮在完整 500 张验证集上计算统一指标。
- 最佳模型按验证集 `u3_aligned_nrmse` 最低选择，即采用 U3Net 的逐图最小值—最大值范围对齐后 NRMSE。
- 三种方法都从头训练。

## 每轮保存内容

每个方法的 `weights/epoch_NNN.pth` 保存该轮模型权重、协议和该轮指标。`last.pth` 额外保存优化器、AMP scaler 和完整历史，用于中断恢复；`best.pth` 指向当前验证最优模型。

`history.csv` 和 `history.json` 每轮记录：训练 loss、全部验证指标、学习率、更新次数、每轮耗时、吞吐率和 CUDA 峰值显存。

训练结束后还会生成 `best_epochs_by_metric.json` 和 `best_epochs_by_metric.csv`。它们为每种方法分别记录每个验证指标的最佳轮次、指标值和对应的逐轮权重路径。MAE、RMSE、NRMSE、PGE 和重缠绕误差取最小值；SSIM 与 AU 指标取最大值。主结果的 `best.pth` 仍固定按 `u3_aligned_nrmse` 选择。

## 输出位置

- DLPU：`experiments/results/gfs128_upstream/runs/dlpu`
- HF：`experiments/results/gfs_rme128/runs/GFS128/hf_matched`
- WWFCA：`experiments/results/gfs_rme128/runs/GFS128/wwfca_only`
- 汇总与图片：`experiments/results/gfs128_dlpu_hf_wwfca`
