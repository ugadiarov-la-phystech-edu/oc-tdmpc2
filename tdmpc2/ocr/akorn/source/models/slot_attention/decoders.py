from typing import List, Optional, Tuple, Dict, Type

import torch
from torch import nn

from ocr.akorn.source.models.slot_attention import networks


class MLPDecoder(nn.Module):
    """Decoder that reconstructs independently for every position and slot."""

    def __init__(
        self,
        inp_dim: int,
        outp_dim: int,
        hidden_dims: List[int],
        n_patches: int,
        activation: Type[nn.Module] = nn.ReLU,
        eval_output_size: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.outp_dim = outp_dim
        self.n_patches = n_patches
        self.eval_output_size = list(eval_output_size) if eval_output_size else None

        self.mlp = networks.MLP(inp_dim, outp_dim + 1, hidden_dims, activation=activation)
        self.pos_emb = nn.Parameter(torch.randn(1, 1, n_patches, inp_dim) * inp_dim**-0.5)

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        bs, n_slots, dims = slots.shape

        if not self.training and self.eval_output_size is not None:
            assert False
            pos_emb = timm.layers.pos_embed.resample_abs_pos_embed(
                self.pos_emb.squeeze(1),
                new_size=self.eval_output_size,
                num_prefix_tokens=0,
            ).unsqueeze(1)
        else:
            pos_emb = self.pos_emb

        slots = slots.view(bs, n_slots, 1, dims).expand(bs, n_slots, pos_emb.shape[2], dims)
        slots = slots + pos_emb

        recons, alpha = self.mlp(slots).split((self.outp_dim, 1), dim=-1)

        masks = torch.softmax(alpha, dim=1)
        recon = torch.sum(recons * masks, dim=1)

        return {"reconstruction": recon, "masks": masks.squeeze(-1)}
