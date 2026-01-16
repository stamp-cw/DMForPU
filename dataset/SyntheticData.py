import torch
from torch.utils.data import Dataset, DataLoader

def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    """Wrap continuous phase to [-pi, pi]."""
    return torch.atan2(torch.sin(phi), torch.cos(phi))

def synthesize_pair(image_size: int, device: torch.device):
    """Generate a synthetic unwrapped/ wrapped phase pair for quick tests."""
    # Create a smooth ramp with sine perturbations as unwrapped phase
    x = torch.linspace(-torch.pi, torch.pi, steps=image_size, device=device)
    y = torch.linspace(-torch.pi, torch.pi, steps=image_size, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    unwrapped = grid_x + 0.3 * torch.sin(3 * grid_y) + 0.2 * torch.sin(5 * grid_x)
    wrapped = wrap_phase(unwrapped)
    return unwrapped.unsqueeze(0), wrapped.unsqueeze(0)

class InSARDataset(Dataset):
    """Simple synthetic dataset; replace with real loader for InSAR-DLPU."""

    def __init__(
            self,
            root,
            split='train',
            transform=None,
            target_transform=None,
            joint_transform=None,
            scale_k=5,
            length: int = 32, image_size: int = 64, device = None):
        self.length = length
        self.image_size = image_size
        self.device = device or torch.device("cpu")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        unwrapped, wrapped = synthesize_pair(self.image_size, self.device)
        sample = {
            "wrapped": wrapped.float(),
            "unwrapped": unwrapped.float(),
        }
        return sample