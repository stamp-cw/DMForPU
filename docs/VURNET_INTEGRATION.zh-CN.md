# VUR-Net 集成说明

- 上游仓库：`https://github.com/lyuzinmaxim/VUR-Net`
- 固定提交：`59168a529fe879daeac8fa2f03ea60b91ea4e0b3`
- 网络源文件：`third_party/VUR-Net/VURNet.py`
- 上游源文件修改：否
- 参数量：21,561,430，全部可训练
- 输入与输出：单通道缠绕相位到单通道解缠相位，原网络可直接处理 128×128

## GFS128 训练协议

- 数据：与其他方法相同的 GFS128 固定 4,500/500 训练/验证划分
- epoch：500
- batch size：20
- optimizer：Adam
- learning rate：`1e-4`
- loss：逐像素 L1
- 模型选择：验证集 `mean_aligned_mae` 最低的权重
- 测试：GFS128 的 clean、0、5、10、20、30 dB 条件
- 指标：项目统一的 raw、integer-aligned、mean-aligned、U3-range-aligned、SSIM、AU、PGE 和重新缠绕一致性指标

README 说明原论文采用 Adam、初始学习率 `2e-4` 并在停滞时减半；仓库作者同时记录了 Adam `1e-4` 训练 500 轮的成功设置。本项目采用后者，避免复刻示例 `config.yml` 中仅用于演示的 2 轮配置。

上游 `forward` 包含一条逐批打印张量尺寸的调试语句。项目训练适配器只静默该输出，不改变网络结构、参数或计算。

## 资源检查

RTX 4090、128×128、batch 20 的单步前向和反向检查：

- 输出形状：`[20, 1, 128, 128]`
- 峰值已分配显存：约 2,754 MiB
- 实测训练步耗时：约 0.070 秒

上游仓库当前没有提供 LICENSE 文件，发表或分发其代码前需要向作者确认授权条款。
