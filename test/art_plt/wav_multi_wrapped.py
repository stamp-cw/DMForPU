import os.path

import scipy.io as sio
import torch
import matplotlib.pyplot as plt

from utils.util import multi_scale_wavelet

gt_wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_data/wrapped_000006_part_1_3.mat')['input']
gt_wrapped = torch.as_tensor(gt_wrapped)

# sin_wrapped = multi_scale_wavelet(torch.sin(gt_wrapped), 'db4', level=2)
# print(sin_wrapped.shape)
# # cos_wrapped = multi_scale_wavelet(torch.cos(gt_wrapped), 'db4', level=2)
# sin_wrapped = sin_wrapped.unsqueeze(0)
# b,c,h,w = sin_wrapped.shape
#
# for i in range(c):
#     img_array = sin_wrapped[:,i,:,:]
#     img = img_array.squeeze()  # [H, W]
#     img = img.detach().cpu().numpy()
#     # img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
#     # img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/noisy_t{t}.png")
#     img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/sin_multi_wav_wrapped_{i}.png")
#     plt.figure(figsize=(4, 4))
#     plt.imshow(img, cmap="twilight")
#     plt.axis("off")
#     # plt.colorbar(fraction=0.046, pad=0.04)
#     plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
#     plt.close()

cos_wrapped = multi_scale_wavelet(torch.cos(gt_wrapped), 'db4', level=2)
print(cos_wrapped.shape)
# cos_wrapped = multi_scale_wavelet(torch.cos(gt_wrapped), 'db4', level=2)
cos_wrapped = cos_wrapped.unsqueeze(0)
b,c,h,w =cos_wrapped.shape

for i in range(c):
    img_array = cos_wrapped[:,i,:,:]
    img = img_array.squeeze()  # [H, W]
    img = img.detach().cpu().numpy()
    # img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
    # img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/noisy_t{t}.png")
    img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/cos_multi_wav_wrapped_{i}.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="twilight")
    plt.axis("off")
    # plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
    plt.close()