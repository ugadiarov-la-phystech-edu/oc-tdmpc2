from sympy import prod
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import einops


def reshape(x: torch.Tensor, n: int):
    if x.ndim == 3:  # x.shape = ([B, T, C ])
        return x.transpose(1, 2).unflatten(1, (-1, n))
    else:  # x.shape = ([B, C, ..., ])
        return x.unflatten(1, (-1, n))


def reshape_back(x):
    if x.ndim == 4:  # Tokens
        return x.flatten(1, 2).transpose(1, 2)
    else:
        return x.flatten(1, 2)


def _l2normalize(x):
    return torch.nn.functional.normalize(x, dim=2)


def norm(n, x, dim=2, keepdim=True):
    return torch.linalg.norm(reshape(x, n), dim=dim, keepdim=keepdim)


def normalize(x: torch.Tensor, n):
    x = reshape(x, n)
    x = _l2normalize(x) 
    x = reshape_back(x)
    return x

# currently not used
def compute_exponential_map(n, x, dxdt, reshaped_inputs=False):
    if not reshaped_inputs:
        dxdt = reshape(dxdt, n)
        x = reshape(x, n)
    norm = torch.linalg.norm(dxdt, dim=2, keepdim=True)
    norm = torch.clip(norm, 0, math.pi)
    nx = torch.cos(norm) * x + torch.sin(norm) * (dxdt / (norm + 1e-5))
    if not reshaped_inputs:
        nx = reshape_back(nx)
    return nx


class Normalize(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return normalize(self.n, x)
