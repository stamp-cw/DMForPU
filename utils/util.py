from collections import defaultdict

import torch

import torch.nn.functional as F

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap continuous phase to [-pi, pi]."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))

    # psi = (phi + torch.pi) % (2 * torch.pi) - torch.pi
    #
    # return psi


def median_filter2d(input, kernel_size=3):
    # 确保 kernel_size 为奇数
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    padding = kernel_size // 2

    # 对每个通道和每个批次进行处理
    # 使用滑动窗口计算局部区域的中位数
    unfolded = F.unfold(input, kernel_size=kernel_size, padding=padding)
    unfolded = unfolded.view(input.size(0), input.size(1), kernel_size * kernel_size, -1)

    # 对每个窗口进行排序，并取中位数
    unfolded, _ = unfolded.sort(dim=2)
    median = unfolded[:, :, kernel_size * kernel_size // 2, :].view_as(input)

    return median



class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics: dict):
        for k, v in metrics.items():
            self.sums[k] += v
            self.counts[k] += 1

    def avg(self):
        return {k: self.sums[k] / self.counts[k] for k in self.sums}