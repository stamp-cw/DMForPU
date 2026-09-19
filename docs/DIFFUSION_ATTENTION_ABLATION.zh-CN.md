# HF diffusion 与 WWFCA 的注意力消融边界

## HF diffusion 基线

HF diffusion 的去噪骨干使用 Hugging Face `UNet2DModel`：

- 所有下采样块均为 `DownBlock2D`。
- 所有上采样块均为 `UpBlock2D`。
- `add_attention=False`。
- 不包含 `CrossAttnDownBlock2D`、`CrossAttnUpBlock2D` 或 cross-attention 参数。
- 缠绕相位通过通道拼接与扩散状态共同输入网络。
- 扩散参数化保持 `prediction_type="sample"`，即预测干净相位 `x0`。

当前三层 32/64/64 基线参数量为 1,046,017。

## DCC × WWFCA 配对消融

| 方法 | DCC 条件 | WWFCA 交叉注意力 | 默认参数量 |
|---|---:|---:|---:|
| `hf_matched` | 否 | 否 | 11,334,913 |
| `dcc_only` | 是 | 否 | 11,336,065 |
| `wwfca_only` | 否 | 是 | 13,908,481 |
| `fdu` / DCC+WWFCA | 是 | 是 | 13,909,633 |

`hf_matched` 与 `dcc_only` 使用无注意力的 HF `UNet2DModel`。`wwfca_only` 与 `fdu` 使用项目的 `FDUNet`，其三个自定义交叉注意力模块为：

- `FDUNetMidBlock2DCrossAttn`
- `FDCrossAttnDownBlock2D`
- `FDCrossAttnUpBlock2D`

WWFCA 的交叉注意力在内部以低频 LL 分量作为 query，以 LH、HL、HH 高频分量作为 key/value。传给模型接口的占位 `encoder_hidden_states` 不承担这一频域条件作用。

架构改变后，旧 HF/Matched 权重与新模型不兼容。所有相关方法必须基于新 GFS128 数据从头训练，训练调度不再复用旧 HF 权重。
