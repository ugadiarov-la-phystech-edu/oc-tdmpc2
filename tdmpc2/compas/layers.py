import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from compas.configuration import try_torch_compile


@try_torch_compile
def shared_key_slot_attention(query_slots: torch.Tensor,
                              key: torch.Tensor,
                              scale: float):
    attn = query_slots @ key.transpose(-2, -1) * scale
    attn = F.softmax(attn, dim=1)
    slots = attn @ key

    return slots, attn


class SlotsExtractorBlock(nn.Module):
    def __init__(self, dim: int, feat_dim: int, hidden_mult: int = 4, scale: Optional[int] = None):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, dim, bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        self.scale = scale
        if scale is None:
            self.scale = 1 / math.sqrt(dim)

    def forward_last_timestep(self, slots: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = slots.size(1)
        in_slots = slots[:, -1]
        in_features = features[:, -1]
        query = self.q_proj(in_slots)
        keys = self.k_proj(in_features)
        update, attn_vis = shared_key_slot_attention(query, keys, self.scale)
        new_slots = self.mlp_out(update) + in_slots
        return new_slots, attn_vis

    def forward(self, slots: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = slots.size(1)
        slots = slots.flatten(0, 1)
        features = features.flatten(0, 1)
        query = self.q_proj(slots)
        keys = self.k_proj(features)
        update, attn_vis = shared_key_slot_attention(query, keys, self.scale)
        new_slots = self.mlp_out(update) + slots
        return new_slots.unflatten(0, (-1, T)), attn_vis.unflatten(0, (-1, T))


class TemporalSlotsBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        b = slots.size(0)
        temp_slots = slots.transpose(1, 2).flatten(0, 1)
        T = temp_slots.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=slots.device)
        attn_slots, _ = self.attn(temp_slots, temp_slots, temp_slots, attn_mask=mask, is_causal=True)
        temp_slots = self.mlp_out(attn_slots)
        return temp_slots.unflatten(0, (b, -1)).transpose(1, 2) + slots


class ActionAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, hidden_mult: int = 4):
        super().__init__()
        # self.start_action_token = nn.Parameter(torch.zeros(dim), requires_grad=True)

        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def gen_causal_mask(self, slots: torch.Tensor) -> torch.Tensor:
        time_steps = slots.size(1)
        num_slots = slots.size(2)
        slots_mask = nn.Transformer.generate_square_subsequent_mask(time_steps,
                                                                    device=slots.device,
                                                                    dtype=slots.dtype)
        slots_mask = slots_mask.repeat_interleave(num_slots, 0)
        # slots_mask = slots_mask.repeat_interleave(num_slots, 1)
        return slots_mask

    def forward(self, slots: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        n_slots = slots.size(-2)
        mask = self.gen_causal_mask(slots)
        slots = slots.flatten(1, 2)
        attn, _ = self.attn(slots, actions, actions, attn_mask=mask)
        slots = self.mlp_out(attn)
        return slots.unflatten(1, (-1, n_slots))


class SlotsInteractionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        b = slots.size(0)
        flat_slots = slots.flatten(0, 1)
        T = flat_slots.size(1)

        attn_slots, _ = self.attn(flat_slots, flat_slots, flat_slots)
        flat_slots = self.mlp_out(attn_slots)
        return flat_slots.unflatten(0, (b, -1)) + slots


class SlotsDynamicsBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, hidden_mult: int = 4):
        super().__init__()
        self.dim = dim

        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def gen_causal_mask(self, slots: torch.Tensor) -> torch.Tensor:
        time_steps = slots.size(1)
        num_slots = slots.size(2)
        slots_mask = nn.Transformer.generate_square_subsequent_mask(time_steps,
                                                                    device=slots.device,
                                                                    dtype=slots.dtype)
        slots_mask = slots_mask.repeat_interleave(num_slots, 0)
        slots_mask = slots_mask.repeat_interleave(num_slots, 1)
        return slots_mask

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        T = slots.size(1)
        mask = self.gen_causal_mask(slots)
        flat_slots = slots.flatten(1, 2)
        attn_slots, _ = self.attn(flat_slots, flat_slots, flat_slots, attn_mask=mask)
        flat_slots = self.mlp_out(attn_slots)
        return flat_slots.unflatten(1, (T, -1)) + slots


# class LinearTemporalSlotsBlock(nn.Module):
#     def __init__(self, dim: int, hidden_mult: int = 4, scale: Optional[int] = None):
#         super().__init__()
#         self.scale = scale
#         if scale is None:
#             self.scale = 1 / math.sqrt(dim)
#
#         self.dim = dim
#
#         self.qkv_proj = nn.Linear(dim, dim * 3)
#
#
#         self.mlp_out = nn.Sequential(
#             nn.Linear(dim, dim * hidden_mult),
#             nn.SELU(),
#             nn.Linear(dim * hidden_mult, dim * hidden_mult),
#             nn.SELU(),
#             nn.Linear(dim * hidden_mult, dim),
#         )
#
#     def forward(self, slots: torch.Tensor) -> torch.Tensor:
#         b = slots.size(0)
#         temp_slots = slots.transpose(1, 2).flatten(0, 1)
#         q, k, v = self.qkv_proj(temp_slots).split(self.dim, dim=-1)
#
#         attn_slots = linear_attention(q, k, v, scale=self.scale)
#         temp_slots = self.mlp_out(attn_slots)
#         return temp_slots.unflatten(0, (b, -1)).transpose(1, 2) + slots
#


class SequentialSlotsExtractorBlock(nn.Module):
    def __init__(self, dim: int, feat_dim: int, hidden_mult: int = 4, chunk_size: int = 4, scale: Optional[int] = None):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, dim, bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        self.chunk_size = chunk_size
        self.scale = scale
        if scale is None:
            self.scale = 1 / math.sqrt(dim)


def forward(self, slots: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, n, d = slots.shape

    slots_chunked = torch.split(slots, self.chunk_size, dim=1)
    feats_chunked = torch.split(features, self.chunk_size, dim=1)
    slots_collected = []
    attns = []

    for slots, feats in zip(slots_chunked, feats_chunked):
        slots = slots.flatten(0, 1)
        feats = feats.flatten(0, 1)
        query = self.q_proj(slots)
        keys = self.k_proj(feats)
        update, attn_vis = shared_key_slot_attention(query, keys, self.scale)
        processed_slots = self.mlp_out(update) + slots
        slots_collected.append(processed_slots.unflatten(0, (b, -1)))
        attns.append(attn_vis.unflatten(0, (b, -1)))

    return torch.stack(attns, dim=1), torch.stack(attns, dim=1)


#
class AutoregressiveSlotsExtractorBlock(nn.Module):
    def __init__(self, dim: int, feat_dim: int, hidden_mult: int = 4, scale: Optional[int] = None):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, dim, bias=False)

        self.mlp_out = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim * hidden_mult),
            nn.SELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        self.scale = scale
        if scale is None:
            self.scale = 1 / math.sqrt(dim)

    def forward_single_t(self, slots: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.q_proj(slots)
        keys = self.k_proj(features)
        update, attn_vis = shared_key_slot_attention(query, keys, self.scale)
        slots = self.mlp_out(update) + slots
        return slots, attn_vis

    def forward(self, slots: torch.Tensor, t_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = t_features.size(1)
        attns: List[torch.Tensor] = []
        processed_slots: List[torch.Tensor] = []
        for t in range(T):
            slots, attn_vis = self.forward_single_t(slots, t_features[:, t])
            processed_slots.append(slots)
            attns.append(attn_vis)

        return torch.stack(processed_slots, dim=1), torch.stack(attns, dim=1)


# class AutoregressiveSlotsExtractorBlock(SlotsExtractorBlock):
#     def forward(self, slots: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         query = self.q_proj(slots)
#         keys = self.k_proj(features)
#         update, attn_vis = shared_key_slot_attention(query, keys, self.scale)
#         return self.mlp_out(update) + slots, attn_vis


class AutoregressiveStackSlotsExtractor(nn.Module):
    def __init__(self,
                 dim: int,
                 feat_dim: int,
                 hidden_mult: int = 4,
                 num_layer: int = 4,
                 scale: Optional[int] = None):
        super().__init__()

        self.blocks = nn.ModuleList([
            AutoregressiveSlotsExtractorBlock(dim,
                                              feat_dim,
                                              hidden_mult=hidden_mult,
                                              scale=scale) for _ in range(num_layer)
        ])

    def forward(self, slots: torch.Tensor, t_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = t_features.size(1)
        attns: List[torch.Tensor] = []
        processed_slots: List[torch.Tensor] = []

        for t in range(T):
            for block in self.blocks:
                slots, attn_vis = block.forward_single_t(slots, t_features[:, t])
            processed_slots.append(slots)
            attns.append(attn_vis)

        return torch.stack(processed_slots, dim=1), torch.stack(attns, dim=1)
