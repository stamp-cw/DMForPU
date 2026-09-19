# GFS128：200扩散时间步下的预测目标与方向高频残差对比

## 实验问题

比较四组模型：

| 模型 | 骨干 | 预测目标 |
|---|---|---|
| `hf_x0` | HF UNet | 归一化干净相位 `x0` |
| `directional_x0` | HF UNet＋LH/HL/HH方向卷积残差 | `x0` |
| `hf_epsilon` | HF UNet | 高斯扩散噪声 `epsilon` |
| `directional_epsilon` | HF UNet＋LH/HL/HH方向卷积残差 | `epsilon` |

## 固定协议

- 数据集：GFS128固定4500/500训练验证划分及六种测试条件；
- 扩散训练时间步：200；
- 训练轮数：两个`x0`模型各300轮，两个`epsilon`模型各600轮；
- 四组均从头训练，batch size为8；
- AdamW，weight decay为`1e-4`；
- 前10轮线性预热至`2e-4`，随后余弦衰减至`2e-6`；
- 四组使用相同的数据顺序、时间步和扩散噪声随机种子；
- 每轮保存模型权重及训练指标；
- 每20轮做一次完整500张验证集评测：`x0`固定使用5步，`epsilon`固定使用25步；
- 不搜索推理步数；在各自固定步数下，以验证U3对齐NRMSE选择最佳checkpoint；
- 四个模型依次串行训练，避免单卡多进程竞争降低总吞吐；
- 最终测试串行运行，避免GPU竞争影响推理速度和显存对比。

## 输出

- 每轮权重：`experiments/results/gfs128_t200_prediction_study/runs/<variant>/weights/`；
- 逐轮损失和效率：各variant目录的`history.csv`；
- 固定推理步数和最佳checkpoint：各variant目录的`selection.json`；
- 最终测试和汇总：实验根目录的`summary.json`、`all_test_metrics.csv`和`comparison.png`。
