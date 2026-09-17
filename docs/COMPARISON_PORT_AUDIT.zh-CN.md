# 对比方法移植审查（2026-09-11）

> 后续修复已完成，见 [修复与验证记录](COMPARISON_PORT_FIXES.zh-CN.md)。下文保留审查时的发现，描述的是修复前状态。

审查对象为 `compare_method.md` 中五个仓库，对照当前工作区代码（包括接手前已有的未提交修改）。本次只生成审查材料，没有修改模型、配置或训练实现。问题分为确定的实现错误、官方训练配置差异、评价口径差异。

## 结论

| 方法/模块 | 结果 |
| --- | --- |
| 公共单卡训练器 | **P1：AMP 缩放器丢失，更新流程错误，影响所有启用 AMP 的普通模型** |
| SqdLstm | **P1：转置卷积的空间对齐不等价**；另有初始化差异 |
| U3Net | 核心网络一致；**P1：现有配置把蒸馏推迟到 99,801 轮，多卡流程未接入蒸馏** |
| Uformer | 同选项网络一致；**P2：没有启用官方训练入口的 modulator**，单通道残差也需明确 |
| Restormer | 单通道网络和 L1 损失未发现移植错误 |
| DLPU | 网络、原始弧度输入输出、L1 损失未发现移植错误 |

P1 表示优先修正，会影响训练或方法定义；P2 表示需要明确并修正或披露的差异。未做完整重训练，不能量化各差异造成的最终精度损失。

## 1. 公共 AMP 更新流程错误（P1）

位置：[model/optimizer.py:52](../model/optimizer.py)、[run/train_model.py:173](../run/train_model.py)。

`OptimizerFN.__call__` 接收 `scaler`，调用 `optimization_fn` 时却没有传入。因此 `scaler.scale(loss).backward()` 产生的梯度直接进入裁剪与普通 `optimizer.step()`，没有反缩放，也没有跳过非有限梯度的保护。即使补上传参，裁剪前仍须调用 `scaler.unscale_(optimizer)`。

此外，`GradScaler()` 在每个 batch 重新创建，不能保留动态缩放状态；模型前向和 meter 中大部分损失计算发生在 `autocast()` 外，当前并未正确实现混合精度前向。

**实测**：CUDA 上单参数 SGD，参数初值 1，loss=0.001×参数，lr=0.1，clip=1。正确更新应为 0.9999，当前实现得到 0.9000；scaler 的 step/update 调用次数均为 0。该测试说明更新流程错误，不代表所有 Adam 更新都会按同一倍数偏离。

建议：持久化 scaler；正确设置 autocast 范围；反向后先 unscale，再裁剪，再 scaler.step/update。临时关闭 AMP 可避开这个分支，但不代替修复。

## 2. SqdLstm 转置卷积不是 Keras SAME 的等价实现（P1）

位置：[model/lstm/sqd_lstm.py:55](../model/lstm/sqd_lstm.py)，`u5/u6/u7/u8`。

原版为 Keras `Conv2DTranspose(kernel=3, stride=2, padding='same')`。本地使用 PyTorch `padding=1, output_padding=1`，虽然输出尺寸相同，但裁剪位置不同，导致解码特征相对 skip 特征错位。

**跨框架实测**：2×2 输入左上角为 1，其余为 0，3×3 卷积核为 1…9，无 bias。

```text
TensorFlow/Keras              本地 PyTorch
1 2 3 0                       5 6 0 0
4 5 6 0                       8 9 0 0
7 8 9 0                       0 0 0 0
0 0 0 0                       0 0 0 0
```

最大绝对差为 9。建议对该参数组合使用 `padding=0, output_padding=0` 的完整转置卷积，再裁掉最下方一行和最右方一列，并做逐层对齐测试。不能直接沿用 DLPU 的上采样参数：DLPU 原版自身就是 PyTorch，来源不同。

来源：[官方网络](https://github.com/Laknath1996/DeepPhaseUnwrap/blob/df8bad82cdbde376c3516e8980eca0e0e9f80f75/src/models/architectures.py)。跨框架实验使用 TensorFlow 2.15.1/Keras 2.15；原仓库声明 Keras 2.4.3，因此不是原始软件环境的完整数值复现。

### 初始化差异（P2）

两条 LSTM 保留 PyTorch 默认初始化。原 Keras 使用 Glorot 输入权重、正交循环权重、forget bias=1；当前 forget gate 的两组 bias 之和为随机值。四个转置卷积和输出卷积也没有匹配 Keras 默认的 Glorot/零 bias。普通卷积的 Kaiming normal 与 Keras 截断 He normal 也不是完全相同的采样分布。

这类差异影响从头训练，不意味着已给定相同权重时 LSTM 方程一定不等价。当前纵向序列排列、BN 的 eps/momentum，以及 VAR+0.1×TV 已修正；对应已有测试通过。

## 3. U3Net 蒸馏配置与多卡流程未完成（P1）

位置：[model/u3net_mmodel.py:103](../model/u3net_mmodel.py)、[configs/u3net_synpu_128_big.yaml:28](../configs/u3net_synpu_128_big.yaml)、[run/train_multi_model.py:73](../run/train_multi_model.py)。

单卡包装器已经实现冻结教师、带噪教师输入、原始梯度学生输入、逐阶段梯度蒸馏损失，并在阶段切换时重建优化器。这部分与原版流程相符。

但是四份 `u3net*.yaml` 都是 `brand_new_epochs=100001, distill_epochs=200`，推导出的起始 epoch 为 **99801（从 0 计数）**。按常见的几百轮预算停止，实际上不会进入蒸馏。官方示例是 500 轮自监督+200 轮蒸馏。应使用明确的训练预算或设置 `distill_start_epoch`，并记录最终权重属于哪个阶段。

`train_multi_model.py` 没有调用 `configure_training_phase`，也没有保存教师训练状态。使用该入口时，单卡新增的蒸馏功能不会自动生效。此项为静态调用链检查，未启动多 GPU 实验。

另有训练协议差异：官方例子 lr=1e-3、指数调度 gamma=0.99，本地 lr=1e-4，公共优化器只做 warmup/裁剪；原版使用逐样本 SNR，本地使用统一配置 SNR。固定 SNR 数据集可以合理适配，混合噪声数据则需逐样本条件。

来源：[官方训练](https://github.com/chenzhile1999/Unsupervised-PU/blob/47c0af31fee5ee7f442aa433a5fb3483a4376334/train.py)、[官方数据处理](https://github.com/chenzhile1999/Unsupervised-PU/blob/47c0af31fee5ee7f442aa433a5fb3483a4376334/data.py)。

## 4. Uformer 与官方实际构建参数不同（P2）

位置：[model/transformer/uformer.py:1078](../model/transformer/uformer.py)、[model/model_setup.py:13](../model/model_setup.py)。

官方 `get_arch` 对常用 Uformer/T/S/B 显式设置 `modulator=True`；本地只传 config，构造器默认 `modulator=False`，实测所有块的 modulator 数量为 0。仅在 YAML 添加该字段也不会生效，因为构造器没有读取它。当前结构不能直接称为官方默认完整配置。

此外保留了原代码 `return x+y if self.dd_in==3 else y`。将通道数改成 1 后会同时关闭全局残差。对于相位解缠的绝对相位回归，这可以是有意适配，不宜直接断言必须加残差；应明确选择并记录。官方 B 的层数也是 `[1,2,8,8,2,8,8,2,1]`，当前全为 2，接近 S 的深度，不能标成 B。

来源：[官方构建入口](https://github.com/ZhendongWang6/Uformer/blob/65fc970a8ffc09605faca74ed016ee93c9ad8a36/utils/model_utils.py)、[官方网络](https://github.com/ZhendongWang6/Uformer/blob/65fc970a8ffc09605faca74ed016ee93c9ad8a36/model.py)。

## 5. 评价口径与上次五图对比的限制

SqdLstm 的 VAR+TV 和 U3Net 的梯度损失都不约束全局常数偏移。只消除整数 2π 偏移不能消除任意实数偏移，所以单看绝对 MAE 不能判断移植是否正确。

SqdLstm 官方测试还用真值 min/max 对预测做范围缩放；U3Net 官方指标函数同样包含范围缩放（调用处还存在实参与形参名称顺序相反的情况）。这些口径与本项目原始/整数周期对齐的指标不等价。不可为降低误差而悄悄用真值拟合预测；应同时报告原始误差、明确定义的偏移对齐误差和梯度误差，将原版范围缩放指标单独列出。

本地部分 meter 的 NRMSE 用整个 batch 的极值范围归一化，也不等同于逐样本 NRMSE 再平均。上次比较脚本采用逐样本计算，不能与训练日志的数值直接混用。

来源：[SqdLstm 测试](https://github.com/Laknath1996/DeepPhaseUnwrap/blob/df8bad82cdbde376c3516e8980eca0e0e9f80f75/test_model.py)、[U3Net 指标](https://github.com/chenzhile1999/Unsupervised-PU/blob/47c0af31fee5ee7f442aa433a5fb3483a4376334/metrics.py)。历史预测没有训练配置、代码提交和完整权重溯源，不能将其误差直接归因于本次发现的问题。

## 6. 通过的检查与可复现材料

对四个 PyTorch 原版，在相同选项下严格加载同一 state_dict，以固定随机输入执行 eval 前向：

| 网络 | 参数量 | 最大输出绝对差 |
| --- | ---: | ---: |
| DLPU | 26,114,881 | 0 |
| Restormer（单通道） | 26,124,052 | 0 |
| Uformer（相同单通道选项） | 20,603,835 | 0 |
| U3Net（三阶段） | 742,143 | 0 |

这是 32×32 CPU 数值等价检查；Uformer 窗口为 4，保持上游和本地一致，不是性能评测。其余模块的源码差异也经过检查。已有 `test_comparison_ports.py` 5 项、`test_noise_and_u3net.py` 6 项全部通过；这些测试未覆盖上述新发现，不能据此宣称完整移植正确。

DLPU 来源：[Network.py](https://github.com/kqwang/Phase_unwrapping_by_U-Net/blob/3c84f34846dd8e9fbbdc613d161e278c2279d1bc/Network.py)。Restormer 来源：[restormer_arch.py](https://github.com/swz30/Restormer/blob/68dc6ac472db26f16361150cb7a96a1bc87da93f/basicsr/models/archs/restormer_arch.py)。

审查用原仓库固定副本、`check_ports.py`、`verification.json` 和 TensorFlow 脉冲输出位于 `tmp/upstream_audit/`。TensorFlow 验证环境独立放在 `tmp/audit_tf/`，项目 `.venv` 依赖未改变。

建议顺序：先修公共 AMP 和 SqdLstm 空间对齐，再明确 U3Net 完整预算/蒸馏及 Uformer 变体，最后统一评价协议并重新训练对比权重。
