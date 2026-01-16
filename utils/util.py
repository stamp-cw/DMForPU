from collections import defaultdict

import torch

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap continuous phase to [-pi, pi]."""
    # return torch.atan2(torch.sin(phi), torch.cos(phi))

    psi = (phi + torch.pi) % (2 * torch.pi) - torch.pi

    return psi


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