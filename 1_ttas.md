• 结论：这次拒稿不是方法完全不行，而是“论文的核心主张大于现有证据”。不建议申诉，也不建议
  只改文字后立刻转投。应先补齐物理一致性、真实数据、不确定性和效率四条证据链，再重写论
  文。

  ## 最优先修改

  ### 1. 补相位物理一致性

  审稿人 R3.1、R3.6 是最致命的问题。

  当前代码其实已经计算了 WrappedL1 和 WrappedRMSE，即把预测结果重新缠绕后与输入相位比较，
  见 meter/wav_meter.py:30。但存在两个不足：

  - 论文没有把它作为 cycle consistency 结果正式报告。
  - 当前使用普通 L1/RMSE，在 -π 与 π 边界不严谨，应改成圆周距离。

  建议新增：

  circular_error = atan2(
      sin(wrap(pred) - wrapped_input),
      cos(wrap(pred) - wrapped_input)
  )

  报告：

  - Rewrapping MAE/RMSE
  - 超过 π/10、π/4 的错误像素比例
  - 原始 MAE/RMSE
  - 经过全局 2πk 偏移对齐后的 MAE/RMSE
  - 估计得到的全局整数偏移 k

  如果想进一步强化物理约束，可增加一个很小的 cycle-consistency loss，但必须重新训练并做
  有/无该损失的消融，不能只在论文里声称已有约束。

  ### 2. 真正评估 100 对真实 TanDEM-X 数据

  这是 R2.1、R3.2、R3.3、R3.9 的共同重点。

  本地数据已经包含：

  test_wrapped_real/       100
  test_absolute_real/      100

  但当前 dataset/InSARDLPUMat.py:12 只读取 test_wrapped/ 和 test_absolute/，也就是说这
  100 对真实数据没有进入正式评估流程。

  需要：

  1. 新增 InSARDLPURealDataset 或支持 split="real"。
  2. 注册独立数据名，例如 InSARDLPUMat256Real。
  3. 为 WWFDiff-PU、DLPU、U3Net、Restormer、Uformer 使用完全相同的真实测试集。
  4. 不能在这 100 对数据上训练或调参。
  5. 报告 MAE、RMSE、NRMSE、PGE、SSIM、cycle consistency。
  6. Abstract 和 Conclusion 必须承认当前方法在真实场景的 MAE/PGE 不如 DLPU，除非新实验能
     够改变结论。

  如果真实数据仍然不占优，就把“强泛化、稳定优于其他方法”改成：

  > synthetic data accuracy is improved, while generalization to real interferograms
  > remains limited.

  ### 3. 修复噪声实验再全部重跑

  R2.3 指出噪声模型不清楚。代码显示目前使用的是“加性高斯相位噪声”，不是复数干涉域乘性散斑
  噪声。

  而且 dataset/SyntheticPUMatNoise.py:82 中存在疑似实现错误：

  wrapped_noise = wrapped_norm_noise * (2 * torch.pi) - torch.pi

  噪声本应是零均值，这里的 -π 会给噪声增加固定偏置。至少应检查是否应为：

  noise_phase = wrapped_norm_noise * (2 * torch.pi)
  wrapped_noisy = wrap_phase(wrapped + noise_phase)

  建议把噪声改成显式配置：

  noise:
    type: additive_phase_gaussian
    snr_db: 10

  并增加自动测试，验证：

  - 噪声均值接近 0
  - 实际 SNR 接近配置 SNR
  - 加噪后相位仍在 [-π, π]
  - 固定 seed 可复现

  修复后，0、5、10、20、30 dB 的所有模型都必须重新评估。旧噪声结果不应继续沿用。

  ### 4. 把预测目标统一说清楚

  R3.4 是明确的代码与论文不一致。

  当前主配置使用：

  prediction_type: sample

  在 diffusion/fdu_ddpm_diffusion.py:30 中，这意味着网络直接预测归一化后的 clean sample
  x₀。PureL1 也是将模型输出和真实 x₀ 比较，不是训练噪声预测器。

  建议：

  - 把变量 noise_pred 重命名为 model_output。
  - 集中实现 build_training_target()。
  - 根据 sample、epsilon、v_prediction 明确返回不同 target。
  - 主论文只描述最终实验实际采用的 sample/x₀ prediction。
  - 明确写出：

  x₀ = normalized ground-truth continuous/unwrapped phase
  c  = [sin(wrapped phase), cos(wrapped phase)]

  R1.3 关于 Pdata(x) 的问题也会同时解决。

  ## 第二优先级实验

   审稿意见               项目改造
  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   R1.1、R3.5 不确定性    新增同一输入重复采样 K=20 或 50 次，保存均值、标准差、绝对误
                          差；计算 uncertainty-error Spearman 相关、风险覆盖曲线
  ─────────────────────  ────────────────────────────────────────────────────────────────
   R1.2、R2.2 计算代价    新增统一 benchmark，报告参数量、FLOPs、单图延迟、吞吐量、峰值
                          显存；扩散模型必须计入完整 5 步
  ─────────────────────  ────────────────────────────────────────────────────────────────
   R3.7 传统方法          加入 MCF、quality-guided、tree/SNAPHU 类方法，使用相同输入、裁
                          剪和指标
  ─────────────────────  ────────────────────────────────────────────────────────────────
   R3.8 相干性            如果数据有 coherence map，按相干性分箱评估并尝试作为额外条件；
                          没有数据就删除“低相干区域鲁棒”主张
  ─────────────────────  ────────────────────────────────────────────────────────────────
   R1.4 DCC 异常          至少用 3 个随机种子重跑；按高/低梯度区统计误差，并画误差分布而
                          不只报告均值
  ─────────────────────  ────────────────────────────────────────────────────────────────
   R1.5、R2.4 可视化      增加高梯度区域 1D 剖面、噪声伪影图、频谱/梯度保持结果，讨论过
                          平滑

  不确定性实验要接受一种可能：如果多次采样几乎没有差异，或者方差与误差不相关，就不要继续
  把“one-to-many uncertainty modeling”作为核心贡献。可以将方法重新定位为“迭代条件恢复”，
  把不确定性写成尚未实现的潜力。

  ## 建议实施顺序

  1. 修复噪声实现、普通模型注册表和评估指标。
  2. 建立统一 evaluate.py，所有方法只走一套数据与指标代码。
  3. 接入真实 TanDEM-X 100 对数据。
  4. 加 cycle consistency 和全局偏移评估。
  5. 加统一效率 benchmark。
  6. 加传统方法基线。
  7. 加重复采样不确定性实验。
  8. 重做 DCC 消融、高梯度剖面和噪声可视化。
  9. 最后重写摘要、方法、实验和结论。
  10. 统一参考文献格式，给扩散/SDE方程补原始引用，压缩 related work。

  这份拒稿意见的任务模式属于 triage-only。目前状态是 blocked for resubmission：在真实数
  据、物理一致性和噪声实现三个问题解决前，不建议转投。仓库中也没有论文正文源文件，因此现
  在只能准确规划代码和实验；如果你把论文 .docx 或 .tex 放进项目，我可以进一步把每条意见映
  射到具体章节、公式、表格和待重写段落。
