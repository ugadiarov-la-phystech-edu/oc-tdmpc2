from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init
from sold.modeling.sold.dynamics import make_ocvp_seq_dynamics_model
from sold.modeling.sold.prediction import Predictor


class SoldWorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            raise ValueError(f'Multitasking is not supported by {self.__class__}')

        self._encoder = layers.enc(cfg)
        sequence_length = cfg.sold_dynamics_num_context + cfg.horizon
        self._dynamics = make_ocvp_seq_dynamics_model(
            num_slots=cfg.n_slots, slot_dim=cfg.slot_dim, sequence_length=sequence_length, action_dim=cfg.action_dim,
            token_dim=cfg.sold_dynamics_token_dim, hidden_dim=cfg.sold_dynamics_hidden_dim, num_layers=cfg.sold_dynamics_num_layers,
            num_heads=cfg.sold_dynamics_num_heads, residual=cfg.sold_dynamics_residual, input_buffer_size=sequence_length,
            teacher_forcing=cfg.sold_dynamics_teacher_forcing,
        )

        self._reward = Predictor(
            max_episode_steps=cfg.time_limit, num_slots=cfg.n_slots, slot_dim=cfg.slot_dim, token_dim=cfg.sold_reward_token_dim,
            num_heads=cfg.sold_reward_num_heads, num_layers=cfg.sold_reward_num_layers, hidden_dim=cfg.sold_reward_hidden_dim,
            output_dim=max(cfg.num_bins, 1), num_mlp_layers=cfg.sold_reward_num_mlp_layers, action_dim=cfg.action_dim,
        )

        self._pi = Predictor(
            max_episode_steps=cfg.time_limit, num_slots=cfg.n_slots, slot_dim=cfg.slot_dim, token_dim=cfg.sold_actor_token_dim,
            num_heads=cfg.sold_actor_num_heads, num_layers=cfg.sold_actor_num_layers, hidden_dim=cfg.sold_actor_hidden_dim,
            output_dim=2 * cfg.action_dim, num_mlp_layers=cfg.sold_actor_num_mlp_layers,
        )

        self._Qs = nn.ModuleList([Predictor(
            max_episode_steps=cfg.time_limit, num_slots=cfg.n_slots, slot_dim=cfg.slot_dim, token_dim=cfg.sold_reward_token_dim,
            num_heads=cfg.sold_reward_num_heads, num_layers=cfg.sold_reward_num_layers, hidden_dim=cfg.sold_reward_hidden_dim,
            output_dim=max(cfg.num_bins, 1), num_mlp_layers=cfg.sold_reward_num_mlp_layers, action_dim=cfg.action_dim,
        ) for _ in range(cfg.num_q)])

        self.apply(init.weight_init)
        init.zero_([self._reward.mlp[-1].weight] + [q.mlp[-1].weight for q in self._Qs])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.lower_bound = torch.as_tensor(cfg.action_lower_bound, dtype=torch.float32)
        self.upper_bound = torch.as_tensor(cfg.action_upper_bound, dtype=torch.float32)
        self.min_std = torch.tensor(cfg.sold_min_std, dtype=torch.float32)
        self.max_std = torch.tensor(cfg.sold_max_std, dtype=torch.float32)
        self.init_std = torch.tensor(cfg.sold_init_std, dtype=torch.float32)

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        if self.cfg.multitask:
            self._action_masks = self._action_masks.to(*args, **kwargs)
        self.lower_bound = self.lower_bound.to(*args, **kwargs)
        self.upper_bound = self.upper_bound.to(*args, **kwargs)
        self.min_std = self.min_std.to(*args, **kwargs)
        self.max_std = self.max_std.to(*args, **kwargs)
        self.init_std = self.init_std.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        """
        Overriding `train` method to keep target Q-networks in eval mode.
        """
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def track_q_grad(self, mode=True):
        """
        Enables/disables gradient tracking of Q-networks.
        Avoids unnecessary computation during policy optimization.
        This method also enables/disables gradients for task embeddings.
        """
        for p in self._Qs.parameters():
            p.requires_grad_(mode)
        if self.cfg.multitask:
            for p in self._task_emb.parameters():
                p.requires_grad_(mode)

    def soft_update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_target.data.lerp_(p.data, self.cfg.tau)

    def task_emb(self, x, task):
        """
        Continuous task embedding for multi-task experiments.
        Retrieves the task embedding for a given task ID `task`
        and concatenates it to the input `x`.
        """
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs, task):
        """
        Encodes an observation into its latent representation.
        This implementation assumes a single state-based observation.
        """
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == 'rgb' and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task, steps=None, num_context=None):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        if steps is None:
            steps = a.shape[1] - z.shape[1] + 1

        if num_context is None:
            num_context = z.shape[1]

        return self._dynamics.predict_slots(z, a, steps=steps, num_context=num_context)

    def reward(self, z, a, task, start=0):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        return self._reward(z, start=start, actions=a)

    def pi(self, z, task, start=None):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        if start is None:
            start = z.size()[1] - 1

        # Gaussian policy prior
        mu, std = self._pi(z, start=start).squeeze(1).chunk(2, dim=-1)
        mu = torch.tanh(torch.clamp(mu, self.lower_bound, self.upper_bound))
        std = (self.max_std - self.min_std) * torch.sigmoid(std + self.init_std) + self.min_std
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, torch.log(std), size=action_dims)
        pi = mu + eps * std
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, None

    def Q(self, z, a, task, return_type='min', target=False, start=None, **kwargs):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        if start is None:
            start = z.shape[1] - 1

        Qs = self._target_Qs if target else self._Qs
        out = torch.stack([q(z, start=start, actions=a) for q in Qs]).squeeze(2)

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2