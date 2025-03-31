import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import einops
import numpy as np

from ocr.akorn.source.layers.gta import (
    make_2dcoord,
    make_SO2mats,
    rep_mul_x,
)


class Interpolate(nn.Module):

    def __init__(self, r, mode="bilinear"):
        super().__init__()
        self.r = r
        self.mode = mode

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.r, mode=self.mode, align_corners=False
        )


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ResBlock(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ReadOutConv(nn.Module):
    def __init__(
        self,
        inch,
        outch,
        out_dim,
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.outch = outch
        self.out_dim = out_dim
        self.invconv = nn.Conv2d(
            inch,
            outch * out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bias = nn.Parameter(torch.zeros(outch))

    def forward(self, x):
        x = self.invconv(x).unflatten(1, (self.outch, -1))
        x = torch.linalg.norm(x, dim=2) + self.bias[None, :, None, None]
        return x


class BNReLUConv2d(nn.Module):

    def __init__(
        self,
        inch,
        outch,
        kernel_size=1,
        stride=1,
        padding=0,
        norm=None,
        act=nn.ReLU(),
    ):
        super().__init__()
        if norm == "gn":
            norm = lambda ch: nn.GroupNorm(8, ch)
        elif norm == "bn":
            norm = lambda ch: nn.BatchNorm2d(ch)
        elif norm == None:
            norm = lambda ch: nn.Identity()
        else:
            raise NotImplementedError

        conv = nn.Conv2d(
            inch,
            outch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.fn = nn.Sequential(
            norm(inch),
            act,
            conv,
        )

    def forward(self, x):
        return self.fn(x)


class FF(nn.Module):

    def __init__(
        self,
        inch,
        outch,
        hidch=None,
        kernel_size=1,
        stride=1,
        padding=0,
        norm=None,
        act=nn.ReLU(),
    ):
        super().__init__()
        if hidch is None:
            hidch = 4 * inch
        self.fn = nn.Sequential(
            BNReLUConv2d(
                inch,
                hidch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                act=act,
            ),
            BNReLUConv2d(
                hidch,
                outch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm=norm,
                act=act,
            ),
        )

    def forward(self, x):
        x = self.fn(x)
        return x


class LayerNormForImage(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        # x shape: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        if x.ndim == 2:
            x_normalized = self.gamma[..., 0, 0] * x_normalized + self.beta[..., 0, 0]
        else:
            x_normalized = self.gamma * x_normalized + self.beta
        return x_normalized


class ScaleAndBias(nn.Module):
    def __init__(self, num_channels, token_input=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.token_input = token_input

    def forward(self, x):
        # Determine the shape for scale and bias based on input dimensions
        if self.token_input:
            # token input
            shape = [1, 1, -1]
            scale = self.scale.view(*shape)
            bias = self.bias.view(*shape)
        else:
            # image input
            shape = [1, -1] + [1] * (x.dim() - 2)
            scale = self.scale.view(*shape)
            bias = self.bias.view(*shape)
        return x * scale + bias


class RGBNormalize(nn.Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.mean = torch.tensor(mean).view(1, len(mean), 1, 1)
        self.std = torch.tensor(std).view(1, len(std), 1, 1)

    def forward(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x - self.mean) / self.std

    def inverse(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return (x * self.std) + self.mean


class FeatureAttention(nn.Module):
    def __init__(self, n, ch):
        super().__init__()
        self.n = n
        self.ch = ch
        self.q_linear = nn.Linear(n, n)
        self.k_linear = nn.Linear(n, n)
        self.v_linear = nn.Linear(n, n)
        self.o_linear = nn.Linear(n, n)

    def forward(self, x):
        B = x.shape[0]
        q, k, v = map(lambda x: x.view(B, -1, self.n), (x, x, x))
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        o = F.scaled_dot_product_attention(q, k, v)
        return self.o_linear(o).view(B, -1)


class Attention(nn.Module):
    def __init__(
        self,
        ch,
        heads=8,
        weight="conv",
        kernel_size=1,
        stride=1,
        padding=0,
        gta=False,
        rope=False,
        hw=None,
    ):
        super().__init__()

        self.heads = heads
        self.head_dim = ch // heads
        self.weight = weight
        self.stride = stride

        if weight == "conv":
            self.W_qkv = nn.Conv2d(
                ch,
                3 * ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.W_o = nn.Conv2d(
                ch,
                ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif weight == "fc":
            self.W_qkv = nn.Linear(ch, 3 * ch)
            self.W_o = nn.Linear(ch, ch)
        else:
            raise ValueError("weight should be 'conv' or 'fc': {}".format(weight))

        self.gta = gta
        self.rope = rope
        assert (int(self.gta) + int(self.rope)) <= 1  # either gta or rope

        self.hw = hw

        if gta or rope:
            assert hw is not None
            F = self.head_dim // 4
            if self.head_dim % 4 != 0:
                F = F + 1

            if not isinstance(hw, list):
                coord = hw
                _mat = make_SO2mats(coord, F).flatten(1, 2)  # [h*w, head_dim/2, 2, 2]
            else:
                coord = make_2dcoord(hw[0], hw[1])
                _mat = (
                    make_SO2mats(coord, F).flatten(2, 3).flatten(0, 1)
                )  # [h*w, head_dim/2, 2, 2]

            _mat = _mat[..., : self.head_dim // 2, :, :]
            # set indentity matrix for additional tokens

            if gta:
                self.mat_q = nn.Parameter(_mat)
                self.mat_k = nn.Parameter(_mat)
                self.mat_v = nn.Parameter(_mat)
                self.mat_o = nn.Parameter(_mat.transpose(-2, -1))
            elif rope:
                self.mat_q = nn.Parameter(_mat)
                self.mat_k = nn.Parameter(_mat)

    def rescale_gta_mat(self, mat, hw):
        # _mat = [h*w, head_dim/2, 2, 2]
        if hw[0] == self.hw[0] and hw[1] == self.hw[1]:
            return mat
        else:
            f, c, d = mat.shape[1:]
            mat = einops.rearrange(
                mat, "(h w) f c d -> (f c d) h w", h=self.hw[0], w=self.hw[1]
            )
            mat = F.interpolate(mat[None], size=hw, mode="bilinear")[0]
            mat = einops.rearrange(mat, "(f c d) h w -> (h w) f c d", f=f, c=c, d=d)
            return mat

    def forward(self, x):

        if self.weight == "conv":
            h, w = x.shape[2] // self.stride, x.shape[3] // self.stride
        else:
            h, w = self.hw

        reshape_str = (
            "b (c nh) h w -> b nh (h w) c"
            if self.weight == "conv"
            else "b k (c nh)  -> b nh k c"
        )
        dim = 1 if self.weight == "conv" else 2
        q, k, v = self.W_qkv(x).chunk(3, dim=dim)
        q, k, v = map(
            lambda x: einops.rearrange(x, reshape_str, nh=self.heads),
            (q, k, v),
        )
        if self.gta:
            q, k, v = map(
                lambda args: rep_mul_x(self.rescale_gta_mat(args[0], (h, w)), args[1]),
                ((self.mat_q, q), (self.mat_k, k), (self.mat_v, v)),
            )
        elif self.rope:
            q, k = map(
                lambda args: rep_mul_x(args[0], args[1]),
                ((self.mat_q, q), (self.mat_k, k)),
            )

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=self.mask if hasattr(self, "mask") else None
        )

        if self.gta:
            x = rep_mul_x(self.rescale_gta_mat(self.mat_o, (h, w)), x)

        if self.weight == "conv":
            x = einops.rearrange(x, "b nh (h w) c -> b (c nh) h w", h=h, w=w)
        else:
            x = einops.rearrange(x, "b nh k c -> b k (c nh)")

        x = self.W_o(x)

        return x
