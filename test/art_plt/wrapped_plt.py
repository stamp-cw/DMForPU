import os.path

import scipy.io as sio
import torch
import matplotlib.pyplot as plt

gt_wrapped = sio.loadmat('/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_data/wrapped_000006_part_1_3.mat')['input']
gt_wrapped = torch.as_tensor(gt_wrapped).unsqueeze(0).unsqueeze(0)

# img = gt_wrapped.squeeze()  # [H, W]
# img = img.detach().cpu().numpy()
# # img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
# # img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/noisy_t{t}.png")
# img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/wrapped.png")
# plt.figure(figsize=(4, 4))
# plt.imshow(img, cmap="twilight")
# plt.axis("off")
# # plt.colorbar(fraction=0.046, pad=0.04)
# plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
# plt.close()

# gt_sin_wrapped = torch.sin(gt_wrapped)
# img = gt_sin_wrapped.squeeze()  # [H, W]
# img = img.detach().cpu().numpy()
# # img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
# # img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/noisy_t{t}.png")
# img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/sin_wrapped.png")
# plt.figure(figsize=(4, 4))
# plt.imshow(img, cmap="twilight")
# plt.axis("off")
# # plt.colorbar(fraction=0.046, pad=0.04)
# plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
# plt.close()


gt_cos_wrapped = torch.cos(gt_wrapped)
img = gt_cos_wrapped.squeeze()  # [H, W]
img = img.detach().cpu().numpy()
# img_path = self.config.io.generated_sample_png_file_path(self.saved_samples + 1 + i)
# img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/noisy_t{t}.png")
img_path = os.path.join(f"/home/lbxu/xiangyu.liu/stamp-cw/project/DMForPU/test/art_res/cos_wrapped.png")
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="twilight")
plt.axis("off")
# plt.colorbar(fraction=0.046, pad=0.04)
plt.savefig(img_path, bbox_inches="tight", pad_inches=0)
plt.close()

