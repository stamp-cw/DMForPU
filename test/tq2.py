import scipy.io as sio
# import torch
import matplotlib.pyplot as plt
import numpy as np

# wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Test/test_in/000102.mat')['input']
# unwrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/SyntheticPUMat128Test/test_gt/000102.mat')['gt']


wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat256Small/train_wrapped/000101.mat')['input']
unwrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/data/InSARDLPUMat256Small/train_absolute/000101.mat')['output']


cc_w = np.mod((np.pi + unwrapped), 2 * np.pi)
print(f"cc_w [1,1]: {cc_w[1,1]}, wrapped [1,1]: {wrapped[1,1]}")


cc2_w = np.mod((np.pi + unwrapped), 2 * np.pi) - np.pi
print(f"cc2_w [1,1]: {cc2_w[1,1]}, wrapped [1,1]: {wrapped[1,1]}")

diff = unwrapped - wrapped
k = diff // (2 * np.pi)
k_float = diff / (2 * np.pi)
print(f"k [1,1]: {k[1,1]}, wrapped [1,1]: {diff[1,1]}")
print(f"k float [1,1]: {k_float[1,1]}, wrapped [1,1]: {diff[1,1]}")
print(f"k [2,2]: {k[2,2]}, wrapped [1,1]: {diff[2,2]}")
print(f"k float [2,2]: {k_float[2,2]}, wrapped [1,1]: {diff[2,2]}")
print(f"k float [2,2]: {k_float[3,3]}, wrapped [1,1]: {diff[3,3]}")
print(f"k float [2,2]: {k_float[4,4]}, wrapped [1,1]: {diff[4,4]}")

print(np.min(k_float))
