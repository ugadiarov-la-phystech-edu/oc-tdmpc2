from typing import Optional

import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize


class FixedLearnedInit(nn.Module):
    """Learned initialization with a fixed number of slots."""

    def __init__(
        self, n_slots: int, dim: int, initial_std: Optional[float] = None, normalize_slots : bool = False,
            frozen: bool = False
    ):
        super().__init__()
        self.n_slots = n_slots
        self.dim = dim
        self.num_slots = self.n_slots
        self.slot_dim = self.dim
        self.normalize_slots = normalize_slots
        if initial_std is None:
            initial_std = dim**-0.5
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim) * initial_std)
        if frozen:
            self.slots.requires_grad_(False)

        if self.normalize_slots:
            parametrize.register_parametrization(self, 'slots', L2NormParam())

    def forward(self, batch_size: int):
        return self.slots.expand(batch_size, -1, -1)


class L2NormParam(nn.Module):
    def forward(self, x):
        return nn.functional.normalize(x, dim=-1)
