# GFS128 / RME128 从头训练与测试协议

## 数据划分

- 每个数据族的 `train.h5` 固定按随机种子 42 拆为 4,500 张训练和 500 张验证。
- `test_0dB/test_5dB/test_10dB/test_20dB/test_30dB` 各 1,000 张，只用于最终测试。
- GFS128 与 RME128 分别从头训练模型，不共享权重；旧 SyntheticPUMat 权重不加载。

## 训练预算

| 方法 | 预算 |
|---|---:|
| DLPU | 100 epochs |
| PUNet | 300 epochs |
| U3Net | 500 epochs self-recovery + 200 epochs distillation |
| HF diffusion baseline | 300 epochs |
| Chen-full HF diffusion | 210 epochs base/SR + 90 epochs self-distillation |
| SQD-LSTM | 100 epochs |
| Restormer | exactly 300,000 optimizer updates（约 267 epochs） |
| Uformer | 250 epochs |
| Itoh / quality-guided MST / least-squares | 无训练 |

U3Net 直接读取每张图的 SNR，并使用同标准差的新噪声完成配对再扰动。HF baseline 与 Chen-full 使用相同 Hugging Face `UNet2DConditionModel`、DDPM 调度、batch、优化器、训练轮数和五步采样；Chen-full 额外启用物理迭代、CAM、稀疏误差、SR 和 SD。

## 选模和指标

模型只按 500 张验证集的 mean-aligned MAE 选择。U3Net 只从第 501–700 轮蒸馏阶段选权重，Chen-full 只从蒸馏阶段的验证轮次选权重。测试报告 raw、整数 2π 对齐和去均值对齐的 MAE/RMSE、NRMSE、PGE、重缠绕圆周误差。

逐轮保存模型权重、loss、验证指标、更新次数、训练速度、耗时和峰值显存。完整队列结束后统一生成 `comparison.csv/json`、训练曲线、SNR 曲线和 0/10/30 dB 定性误差图。
