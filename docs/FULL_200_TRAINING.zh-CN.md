# SyntheticPUMat128Big 完整数据 200 轮训练

本任务从头训练 PUNet、SQD-LSTM、U3Net，各 200 轮，随机种子固定为 42。

## 数据协议

- 训练：`train_in/train_gt` 全部 20,000 对，不从训练集扣除验证样本。
- 验证：官方测试集按种子 42 固定抽取 500 对，只用于逐轮指标和最佳权重选择。
- 测试：官方测试集余下的 1,500 对，只在训练结束后对验证集最佳权重评估一次。
- 运行前检查全部 22,000 对文件的配对、MAT 键、128×128 形状、有限值、范围和重缠绕一致性，并生成带 SHA-256 的只读 NPY 缓存。

## 训练协议

- PUNet：Adam，L1 损失，初始学习率 `1e-4`，权重衰减 `1e-4`，2 轮线性预热，梯度裁剪 1.0。
- SQD-LSTM：Adam，`VAR + 0.1 TV`，其余设置与 PUNet 相同。
- U3Net：Adam，自监督展开损失；按原 500+200 轮比例缩放为 143 轮自恢复和 57 轮冻结教师蒸馏。进入蒸馏阶段时冻结教师并重置优化器；两个阶段均从 `1e-3` 开始按 `0.99^epoch` 衰减。固定训练噪声条件为项目配置的 30 dB。
- CUDA BF16 AMP（不做梯度缩放），批量 32。BF16 保留 FP32 的指数范围，可避免已在 FP16 压力测试中捕获的梯度溢出。每轮若出现非有限损失、输出、梯度，错误形状或跳过更新，任务立即失败并保留状态。

## 输出

输出目录为 `experiments/results/full_200`：

- `manifest.json`、`cache/`：数据划分、全量审计和缓存校验值。
- `smoke_test.json`：三种模型的前向、反向、更新及指标冒烟测试。
- `runs/<method>/weights/epoch_001.pth ... epoch_200.pth`：每轮模型权重。
- `runs/<method>/history.csv` 和 `history.json`：每轮 loss、训练诊断、全部验证指标、学习率、耗时、吞吐和显存峰值。
- `runs/<method>/last.pth`：包含模型、优化器、AMP scaler、U3Net 教师和历史的精确断点。
- `runs/<method>/best.pth`：按验证集对齐 MAE 选择的权重。
- `runs/<method>/complete.json`：最终测试指标、参数量、权重大小、延迟、显存和 FLOPs。
- `run.log`、`queue_status.json`：总队列日志和状态。

启动命令：

```powershell
uv run python -B -u experiments/full_200_study.py queue --epochs 200 --batch-size 32
```
