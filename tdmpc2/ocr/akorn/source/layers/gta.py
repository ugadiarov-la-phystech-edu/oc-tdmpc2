import torch
import numpy as np
import math
from einops import rearrange


def make_2dcoord(H, W, normalize=False):
    """
    Return(torch.Tensor): 2d coord values of shape [H, W, 2]
    """
    x = np.arange(H, dtype=np.float32)  # [0, H)
    y = np.arange(W, dtype=np.float32)  # [0, W)
    if normalize:
        x = x / H
        y = y / W
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    return torch.Tensor(
        np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2)
    )


def make_SO2mats(coord, nfreqs):
    """
    Args:
    coord: [..., 2 or 3]
    freqs: [n_freqs, 2 or 3]
    Return:
    mats of shape [..., n_freqs, (2 or 3), 2, 2]
    """
    dim = coord.shape[-1]
    b = 10000.0
    freqs = torch.exp(torch.arange(0.0, 2 * nfreqs, 2) * -(math.log(b) / (2 * nfreqs)))
    grid_ths = [
        torch.einsum("...i,j->...ij", coord[..., d : d + 1], freqs).flatten(-2, -1)
        for d in range(dim)
    ]

    _mats = [
        [
            torch.cos(grid_ths[d]),
            -torch.sin(grid_ths[d]),
            torch.sin(grid_ths[d]),
            torch.cos(grid_ths[d]),
        ]
        for d in range(dim)
    ]
    mats = [
        rearrange(torch.stack(_mats[d], -1), "... (h w)->... h w", h=2, w=2)
        for d in range(dim)
    ]
    mat = torch.stack(mats, -3)
    return mat


# GTA
@torch.jit.script
def rep_mul_x(rep, x):
    #  rep.shape=[T, F, 2, 2], x.shape=[B, H, T, F*2]
    shape = x.shape
    d = rep.shape[-1]
    return (
        (rep[None, None] * (x.unflatten(-1, (-1, d))[..., None, :])).sum(-1).view(shape)
    )


@torch.jit.script
def rep_mul_qkv(rep, q, k, v):
    return rep_mul_x(rep, q), rep_mul_x(rep, k), rep_mul_x(rep, v)


@torch.jit.script
def rep_mul_qk(rep, q, k):
    return rep_mul_x(rep, q), rep_mul_x(rep, k)


def embed_block_diagonal(M, n):
    """
    Embed a [h*w, d/2, 2, 2] tensor M into a [h*w, d//2n, 4, 4] tensor M'
    with block diagonal structure.

    Args:
    M (torch.Tensor): Tensor of shape [h*w, d/2, 2, 2]
    n (int): Number of blocks to embed into 2nx2n structure

    Returns:
        torch.Tensor: Tensor of shape [h*w, d//2n, 4, 4]
    """
    h_w, d_half, _, _ = M.shape

    # Initialize an empty tensor for the block diagonal tensor M'
    M_prime = torch.zeros((h_w, d_half // n, 4, 4))

    # Embed M into the block diagonal structure of M_prime
    for t in range(h_w):
        for d in range(d_half // n):
            M_prime[t, d] = torch.block_diag(*[M[t, n * d + i] for i in range(n)])
    print(M_prime.shape)
    return M_prime
