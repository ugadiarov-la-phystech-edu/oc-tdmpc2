from typing import Optional, Tuple

import torch
from torch import nn

from compas.layers import SlotsExtractorBlock, TemporalSlotsBlock
from compas.pos_emb import get_sin_pos_enc


class ExtractorModel(nn.Module):
    def __init__(self,
                 num_slots: int,
                 slots_dim: int,
                 max_timestep: int,
                 feat_dim: int,
                 num_patches: int,
                 slots_pos_encoding: nn.Parameter,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 hidden_mult: int = 4, ):
        super().__init__()
        assert num_layers > 0, "Cannot have less than 1 layer"

        self.slots_dim = slots_dim
        self.num_slots = num_slots
        self.max_timestep = max_timestep
        self.feat_dim = feat_dim
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.hidden_mult = hidden_mult
        self.slots_pos_encoding = slots_pos_encoding
        self.feat_proj = nn.Linear(feat_dim, slots_dim, bias=False)
        self.feat_pos_emb = nn.Parameter(torch.randn(1, 1, num_patches, feat_dim) * feat_dim ** -0.5)
        self.slot_token = nn.Parameter(torch.randn(1, 1, 1, slots_dim))
        self.time_pos_encoding = nn.Parameter(
            get_sin_pos_enc(max_timestep, slots_dim), requires_grad=False)

        self.slots_extractor_blocks = nn.ModuleList(
            [SlotsExtractorBlock(slots_dim, feat_dim, hidden_mult) for _ in range(num_layers)])

        self.slot_temporal_blocks = nn.ModuleList(
            [TemporalSlotsBlock(slots_dim, num_heads, hidden_mult) for _ in range(num_layers)]
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(slots_dim, slots_dim * hidden_mult),
            nn.SELU(),
            nn.Linear(slots_dim * hidden_mult, slots_dim * hidden_mult),
            nn.SELU(),
            nn.Linear(slots_dim * hidden_mult, slots_dim),
            nn.LayerNorm(slots_dim),
        )

    def _init_slots(self, batch_size: int, initial_slots: Optional[torch.Tensor] = None):
        slots = initial_slots
        num_gen_timesteps = self.max_timestep

        if initial_slots is not None:
            num_gen_timesteps = self.max_timestep - slots.size(1)

        if num_gen_timesteps > 0:
            token_slots = self.slot_token.expand(batch_size, num_gen_timesteps, self.num_slots, -1)

            token_slots = token_slots + self.slots_pos_encoding.expand(batch_size, num_gen_timesteps, -1, -1)

            if initial_slots is None:
                return token_slots

            return torch.cat((slots, token_slots), dim=1)

        return initial_slots

    def get_start_slots(self, features: torch.Tensor):
        return self.forward(features)


    def get_next_slot(self, next_features: torch.Tensor, prev_slots: torch.Tensor):
        slots, attns = self.forward(next_features.unsqueeze(1), prev_slots, True)
        return slots[:, -1], attns[:, -1]

    def forward_autoregressive(self,
                               features: torch.Tensor,  # B x T x K x D
                               # actions: torch.Tensor,  # B x (T - 1) x D,
                               init_slots: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N, _ = features.shape

        slots = []
        attns = []
        in_slots = init_slots
        num_steps = T - self.max_timestep

        append_all_slots = init_slots is None

        for i in range(num_steps + 1):
            start_step = i
            end_step = start_step + self.max_timestep
            feats = features[:, start_step:end_step]
            extract_only_last_timestep = not (i == 0 and append_all_slots)
            proc_slots, attn_vis = self.forward(feats, in_slots, extract_only_last_timestep)

            if not extract_only_last_timestep:
                slots.append(proc_slots)
                attns.append(attn_vis)
            else:
                slots.append(proc_slots[:, -1].unsqueeze(1))
                attns.append(attn_vis.unsqueeze(1))

            in_slots = proc_slots[:, 1:]

        return torch.cat(slots, dim=1), torch.cat(attns, dim=1)

    def forward(self,
                features: torch.Tensor,
                init_slots: Optional[torch.Tensor] = None,
                extract_only_last_timestep: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = features + self.feat_pos_emb
        B, T, N, _ = features.shape
        time_enc = self.time_pos_encoding[:, -T:]
        keys = keys
        slots = self._init_slots(B, init_slots)
        slots = slots + time_enc.unsqueeze(2).expand(B, -1, self.num_slots, -1)

        blocks = zip(self.slots_extractor_blocks, self.slot_temporal_blocks)
        in_slots = slots
        for extr_blk, temprl_blk in blocks:
            if extract_only_last_timestep:
                new_slots, attn_vis = extr_blk.forward_last_timestep(slots, features)
                slots = torch.cat((slots[:, :-1], new_slots.unsqueeze(1)), dim=1)
            else:
                slots, attn_vis = extr_blk(slots, keys)
            slots = temprl_blk(slots)

        return in_slots + self.mlp_out(slots), attn_vis

    #


class COMPAS(nn.Module):
    def __init__(self,
                 num_slots: int,
                 slots_dim: int,
                 extr_max_timestep: int,
                 feat_dim: int,
                 num_patches: int,
                 num_heads: int = 4,
                 num_extr_layers: int = 3,
                 hidden_mult: int = 4, ):
        super().__init__()

        self.slots_dim = slots_dim
        self.num_slots = num_slots
        self.extr_max_timestep = extr_max_timestep
        self.feat_dim = feat_dim
        self.num_extr_layers = num_extr_layers
        self.num_patches = num_patches
        self.hidden_mult = hidden_mult

        self.slots_pos_emb = nn.Parameter(torch.randn(1, 1, num_slots, slots_dim) * slots_dim ** -0.5)

        self.extractor_model = ExtractorModel(
            num_slots=num_slots,
            slots_dim=slots_dim,
            max_timestep=extr_max_timestep,
            feat_dim=feat_dim,
            num_patches=num_patches,
            num_heads=num_heads,
            num_layers=num_extr_layers,
            hidden_mult=hidden_mult,
            slots_pos_encoding=self.slots_pos_emb,
        )



    def forward_extr_autoregressive(self,
                                    features: torch.Tensor,  # B x T x K x Fd
                                    init_slots: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        b = features.size(0)
        return self.extractor_model.forward_autoregressive(features,
                                                           init_slots=init_slots)