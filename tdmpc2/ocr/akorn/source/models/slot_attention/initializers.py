from typing import Optional

import torch
from torch import nn


class RandomInit(nn.Module):
    """Sampled random initialization for all slots."""

    def __init__(self, n_slots: int, dim: int, initial_std: Optional[float] = None):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(1, 1, dim))
        if initial_std is None:
            initial_std = dim**-0.5
        self.log_std = nn.Parameter(torch.log(torch.ones(1, 1, dim) * initial_std))

    def forward(self, batch_size: int):
        noise = torch.randn(batch_size, self.n_slots, self.dim, device=self.mean.device)
        return self.mean + noise * self.log_std.exp()
