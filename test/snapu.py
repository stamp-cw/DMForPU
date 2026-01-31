import scipy.io as sio
import numpy as np
from spectral.io import envi
import os

# ========================
# 1️⃣ 读取 .mat 文件
# ========================
mat_file = '/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_data/wrapped_000006_part_1_3.mat'
mat = sio.loadmat(mat_file)

# 查看变量名，找到你的相位矩阵名称
print("MAT 文件变量名:", mat.keys())

# 你的相位矩阵在 'input' 变量里
phase = mat['input']  # shape: (rows, cols)

# ========================
# 2️⃣ 确保相位在 [-pi, pi] 范围
# ========================
phase = np.mod(phase + np.pi, 2*np.pi) - np.pi
phase = phase.astype(np.float32)

# ========================
# 3️⃣ 保存为 ENVI 格式
# ========================
# 输出目录
output_dir = '/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/envi_output'
os.makedirs(output_dir, exist_ok=True)

# ENVI 文件名必须以 .hdr 结尾
envi_file = os.path.join(output_dir, 'phase_envi.hdr')

# ENVI 单波段二维矩阵，需要 shape (rows, cols, 1)
phase_envi = phase[:, :, np.newaxis]

# 写 ENVI 文件（会生成 .hdr 和 .dat 两个文件）
envi.save_image(envi_file, phase_envi, dtype=np.float32, force=True, interleave='bil')

print(f"转换完成！生成文件:\n{envi_file}\n对应的 .dat 文件也在同一目录下")
print("可以在 SNAP Desktop 中打开 .hdr 文件进行 SNAPHU 解缠。")
