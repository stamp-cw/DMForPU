# import numpy as np
# from spectral.io import envi
# import scipy.io as sio
#
# # ------------------
# # 1️⃣ 读取解缠相位（ENVI）
# # ------------------
# unwrapped_file = '/home/lbxu/.../unwrapped_phase.hdr'
# unwrapped = envi.open(unwrapped_file).load()[:,:,0]  # shape: (rows, cols)
#
# # ------------------
# # 2️⃣ 读取真实相位
# # ------------------
# gt_mat = sio.loadmat('/home/lbxu/.../ground_truth.mat')
# ground_truth = gt_mat['input']  # shape: (rows, cols)
#
# # ------------------
# # 3️⃣ 计算 RMSE
# # ------------------
# # 只计算有效区域（非 NaN 或 mask 可选）
# mask = ~np.isnan(ground_truth)  # 如果有无效值
# rmse = np.sqrt(np.mean((unwrapped[mask] - ground_truth[mask])**2))
#
# print(f"RMSE = {rmse:.4f} rad")

from spectral.io import envi
import numpy as np
import os

# 参数
input_file = 'unwrapped.bin'
nrows, ncols = 32, 32  # 改成你的矩阵大小
output_hdr = 'unwrapped.hdr'

# 读取裸二进制
unwrapped = np.fromfile(input_file, dtype=np.float32).reshape((nrows, ncols, 1))  # 单波段

# 保存 ENVI 文件（生成 .hdr）
envi.save_image(output_hdr, unwrapped, dtype=np.float32, force=True, interleave='bil')

print(f"生成 ENVI 文件: {output_hdr} + 对应 .dat，SNAP 可以打开")
