# synpu数据集，缠绕相位图和解缠绕相位图的关系

import scipy.io as sio
# import torch
import matplotlib.pyplot as plt
import numpy as np

wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Test/test_in/000101.mat')['input']
unwrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Test/test_gt/000101.mat')['gt']

print(f"wrapped shape: {wrapped.shape}, unwrapped shape: {unwrapped.shape}")
print(f"wrapped max: {wrapped.max()}, wrapped min: {wrapped.min()}")
print(f"unwrapped max: {unwrapped.max()}, unwrapped min: {unwrapped.min()}")

print(f"wrapped [1,1]: {wrapped[1,1]}, unwrapped [1,1]: {unwrapped[1,1]}")

# Verify the relationship between wrapped and unwrapped phase
# wrapped = unwrapped mod 2π
wrapped_calculated = np.mod(unwrapped + np.pi, 2 * np.pi) -  np.pi
# p_wrapped = unwrapped
print(f"wrapped_calculated [1,1]: {wrapped_calculated[1,1]}, wrapped [1,1]: {wrapped[1,1]}")

k = np.round((unwrapped - wrapped) / (2 * np.pi))
# unwrapped_calculated = wrapped + 2 * np.pi * np.round((unwrapped - wrapped) / (2 * np.pi))
print(f"k [1,1]: {k[1,1]}")
unwrapped_calculated = wrapped + 2 * np.pi * k
print(f"unwrapped_calculated [1,1]: {unwrapped_calculated[1,1]}, unwrapped [1,1]: {unwrapped[1,1]}")

x = unwrapped - wrapped
print(f"x [1,1]: {x[1,1]}")
print(f"x_max: {x.max()}, x_min: {x.min()}")

shift_wrapped = wrapped + np.pi
shift_unwrapped = unwrapped + np.pi

wrapped_calculated2 = np.mod(shift_unwrapped, 2 * np.pi)
print(f"wrapped_calculated2 [1,1]: {wrapped_calculated2[1,1]}, shift_wrapped [1,1]: {shift_wrapped[1,1]}")
t_wrapped = wrapped_calculated2 - np.pi
print(f"t_wrapped [1,1]: {t_wrapped[1,1]}, wrapped [1,1]: {wrapped[1,1]}")

print(round(-0.5))


d_unwrapped = unwrapped - (np.min(unwrapped) // (np.pi * 2)) * (np.pi * 2)
wrapped_calculated3 = np.mod(d_unwrapped + np.pi, 2 * np.pi) -  np.pi
# p_wrapped = unwrapped
print(f"wrapped_calculated [1,1]: {wrapped_calculated3[1,1]}, wrapped [1,1]: {wrapped[1,1]}")


cc_w = np.mod((np.pi + unwrapped), 2 * np.pi)
print(f"cc_w [1,1]: {cc_w[1,1]}, wrapped [1,1]: {wrapped[1,1]}")

