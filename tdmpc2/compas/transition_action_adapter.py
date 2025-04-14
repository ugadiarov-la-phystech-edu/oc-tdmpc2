import torch
from torch import nn


class ContinuousActionAdapter(nn.Module):
    def __init__(self, actions_dim: int, projected_actions_dim: int, norm_actions: bool = True):
        super(ContinuousActionAdapter, self).__init__()
        self.projected_actions_dim = projected_actions_dim
        self.use_identity = actions_dim != projected_actions_dim
        self.first_norm_act = norm_actions
        if norm_actions:
            self.norm_act = nn.LayerNorm(actions_dim)
        self.linear = nn.Linear(actions_dim, projected_actions_dim) if self.use_identity else nn.Identity()

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if self.first_norm_act:
            actions = self.norm_act(actions)

        return self.linear(actions)
