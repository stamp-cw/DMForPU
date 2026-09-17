# 对比方法移植修复与验证

本次修复对应 [2026-09-11 审查报告](COMPARISON_PORT_AUDIT.zh-CN.md)，保留原报告作为修复前证据。

## 已完成

| 问题 | 修复 |
| --- | --- |
| AMP 梯度缩放丢失 | `OptimizerFN` 透传 scaler；先 unscale，再裁剪，再 scaler.step/update，非有限梯度会跳过更新。 |
| 每个 batch 重建 scaler | 单卡训练循环持久化 scaler，checkpoint 保存、恢复其状态；autocast 包含模型前向及损失计算。 |
| SqdLstm 上采样错位 | 四层使用完整 3×3、stride=2 转置卷积，再裁掉右侧/底部，匹配 Keras SAME。 |
| SqdLstm 初始化不一致 | 卷积使用截断 He normal；解码上采样和输出层使用 Glorot/零 bias；LSTM 使用 Glorot 输入权重、正交循环权重和 forget bias=1。PyTorch 多出的 bias 固定为零，保留单组可训练偏置。 |
| U3Net 蒸馏推迟 | 四份配置改为总计 700 轮，epoch 500 开始 200 轮蒸馏（epoch 从 0 计数）。 |
| U3Net 多卡没有蒸馏 | 每个进程切换阶段并冻结未包裹 DDP 的教师；重建优化器；checkpoint 保存教师、优化器和 scaler。单卡/多卡新权重均使用无 `module.` 前缀的模型状态，多卡恢复兼容旧前缀。 |
| U3Net 学习率协议 | 使用 Adam、lr=0.001、指数衰减 gamma=0.99；蒸馏阶段重新从初始学习率衰减；关闭原公共 warmup、weight decay 和裁剪，匹配原训练入口。 |
| Uformer 官方选项未生效 | 显式启用 modulator，构造器读取 YAML 的通道数、尺寸、宽度、深度、head、窗口、MLP、modulator 和 residual。 |
| 单通道隐式关闭残差 | `residual` 成为独立选项，当前四份相位回归配置显式设为 true，不再按通道数隐式决定。当前是 S 深度的单通道适配，不标成 B。 |
| NRMSE 用整个 batch 极值 | 六个普通模型 meter 改为逐样本 RMSE/真值范围后平均；单卡训练/验证按样本数加权，避免末尾短 batch 等权；汇总时 detach，防止跨 batch 保留计算图。 |

Accelerate 入口在反向后先反缩放梯度，再调用公共裁剪/优化步骤，保留 Accelerate 自己的 scaler.step。主进程单独验证时暂时解开 DDP 包装，避免验证前向触发其他进程未参与的 DDP 同步。

## 验证结果

- 单元测试：25 项通过，包含梯度反缩放/溢出跳步、autocast 前向、scaler 持久化、Keras 转置卷积脉冲结果、初始化、残差选项、蒸馏边界/学习率重启和逐样本 NRMSE。
- RTX 4090 实际训练小步：DLPU、SqdLstm、Uformer、Restormer、U3Net 均在 `data/SyntheticPUMat128Big/train_in/000001.mat` 的 128×128 输入上完成前向、反向、参数更新，loss 和参数有限。它是功能验证，不是精度评测。
- U3Net 分别调用 epoch 499、500 验证两阶段；未真的训练 500 轮。阶段切换前 lr≈6.63685e-6，切换后 lr=0.001。
- Accelerate 单 GPU：U3Net 两阶段更新及含教师/scaler 的 checkpoint 保存通过。
- 两个 CPU/Gloo 进程：小型替代网络执行真实 DDP 训练循环，跨过 epoch 499/500/501；教师在两进程间一致；checkpoint 恢复教师及优化器成功，恢复后 lr=0.00099。
- 本机仅一张 GPU，未验证双 GPU/NCCL。CPU 集成脚本把 Accelerate 1.15 的带 CUDA device_ids 屏障替换为普通 Gloo 屏障，仍执行真实跨进程同步。

结果位于 `experiments/results/port_fixes/single_gpu.json`、`accelerate_gpu.json`、`ddp_cpu.json`。验证产生的少量 checkpoint 也仅在该结果目录中。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B experiments/verify_port_training.py --output experiments/results/port_fixes/single_gpu.json
.\.venv\Scripts\python.exe -B experiments/verify_port_training.py --accelerate --output experiments/results/port_fixes/accelerate_gpu.json
.\.venv\Scripts\python.exe -B experiments/verify_distributed_ports.py
```

## 使用修复后代码

1. SqdLstm 权重键名/形状仍可加载，但上采样语义已经改变，旧结果不再代表修复版；应从头训练。
2. Uformer 新增调制参数，旧无调制权重不能严格加载到新默认模型。若仅需复核历史输出，使用独立旧版配置设置 `model.modulator: false`、`model.residual: false`。新实验用修复后的配置从头训练。
3. U3Net 原网络结构没有改变，旧模型权重可读取；旧 checkpoint 没有教师状态时无法恢复当时的蒸馏教师。修复版完整训练应从头开始，修复版 checkpoint 可以继续恢复。
4. 本次只修复并做小步验证，没有重新训练对比模型，也没有替换 `article/data` 历史预测。前次五图排名仍是历史混合来源结果。
5. 没有加入基于真值 min/max 的预测范围缩放。NRMSE 仍是无量纲比例，绝对/整数周期对齐误差与官方范围缩放指标应分别报告；固定 SNR 的当前配置仍使用固定噪声条件。

恢复旧试验和启动新试验应使用不同配置/输出目录，避免把语义不同的权重当作同一次训练续跑。
