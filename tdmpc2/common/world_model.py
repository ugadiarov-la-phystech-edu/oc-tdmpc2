from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from common import layers, math, init
from common.layers import mlp
from dlp import ObjectDynamicsDLP
from dlp.policies import EITCritic, EITActor


class WorldModel(nn.Module):
    """
    TD-MPC2 implicit world model architecture.
    Can be used for both single-task and multi-task experiments.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim)
            for i in range(len(cfg.tasks)):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
        self._encoder = layers.enc(cfg)
        latent_dim = self._encoder[cfg.obs](torch.zeros(1, *cfg.obs_shape[cfg.obs], dtype=torch.float32)).size()[1]
        self._dynamics = layers.mlp(latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim], latent_dim,
                                    act=layers.SimNorm(cfg))
        self._reward = layers.mlp(latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                  max(cfg.num_bins, 1))
        self._pi = layers.mlp(latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)
        self._Qs = layers.Ensemble([layers.mlp(latent_dim + cfg.action_dim + cfg.task_dim, 2 * [cfg.mlp_dim],
                                               max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params[-2]])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

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
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
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

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, **kwargs):
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

        z = torch.cat([z, a], dim=-1)
        out = (self._target_Qs if target else self._Qs)(z)

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class OCDynamicsModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.gnn = layers.GNN(input_dim=self.cfg.slot_dim, hidden_dim=self.cfg.latent_dim,
                              action_dim=self.cfg.action_dim, num_objects=self.cfg.n_slots, ignore_action=False,
                              copy_action=True, edge_actions=True, use_interactions=self.cfg.use_interactions)

    def forward(self, slots, action):
        return self.gnn(slots, action)


class OCRewardModel(nn.Module):
    def __init__(self, cfg, **cfg_overrides):
        super().__init__()
        self.cfg = cfg
        self.act = nn.ReLU(inplace=True)
        slot_dim = cfg_overrides.get('slot_dim', self.cfg.slot_dim)
        n_slots = cfg_overrides.get('n_slots', self.cfg.n_slots)
        action_dim = cfg_overrides.get('action_dim', self.cfg.action_dim)
        self.gnn = layers.GNN(input_dim=slot_dim, hidden_dim=self.cfg.latent_dim,
                              action_dim=action_dim, num_objects=n_slots, ignore_action=False,
                              copy_action=True, edge_actions=True, use_interactions=self.cfg.use_interactions)
        self.mlp = nn.Linear(in_features=slot_dim, out_features=max(self.cfg.num_bins, 1))

    def forward(self, slots, action):
        x = self.gnn(slots, action)
        x = self.act(x)
        return self.mlp(x.sum(dim=1))


class OCPolicy(nn.Module):
    def __init__(self, cfg, **cfg_overrides):
        super().__init__()
        self.cfg = cfg
        self.act = nn.ReLU(inplace=True)
        slot_dim = cfg_overrides.get('slot_dim', self.cfg.slot_dim)
        n_slots = cfg_overrides.get('n_slots', self.cfg.n_slots)
        self.gnn = layers.GNN(input_dim=slot_dim, hidden_dim=self.cfg.latent_dim, action_dim=0,
                              num_objects=n_slots, ignore_action=True, copy_action=False, edge_actions=False,
                              use_interactions=self.cfg.use_interactions)
        self.mlp = nn.Linear(in_features=slot_dim, out_features=2 * self.cfg.action_dim)

    def forward(self, slots):
        x = self.gnn(slots, action=None)
        x = self.act(x)
        return self.mlp(x.sum(dim=1))


class OCWorldModel(nn.Module):
    """
    Slot-based object-centric world model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert not cfg.multitask, f'Multitasking is not implemented for slot-based observations.'

        self._encoder = layers.enc(cfg)
        self._dynamics = OCDynamicsModel(self.cfg)
        self._reward = OCRewardModel(self.cfg)
        self._pi = OCPolicy(self.cfg)
        self._Qs = nn.ModuleList([OCRewardModel(self.cfg) for _ in range(cfg.num_q)])
        self.apply(init.weight_init)
        init.zero_([self._reward.mlp.weight] + [q.mlp.weight for q in self._Qs])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

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
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
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
        raise NotImplementedError('Multitasking is not implemented for slot-based observations')

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

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        return self._dynamics(z, a)

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        return self._reward(z, a)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        # Gaussian policy prior
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, **kwargs):
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

        Qs = self._target_Qs if target else self._Qs
        out = torch.stack([q(z, a) for q in Qs])

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class DDLPWorldModel(nn.Module):
    def __init__(self, cfg, ddlp_model: ObjectDynamicsDLP):
        super().__init__()
        self.cfg = cfg
        self.ddlp_model = ddlp_model
        assert not cfg.multitask, f'Multitasking is not implemented for slot-based observations.'

        self._encoder = layers.enc(cfg)
        self._dynamics = nn.Identity()
        particle_fdim = self.ddlp_model.get_dlp_features_dim()
        background_fdim = self.ddlp_model.get_dlp_background_dim()
        self._reward = EITCritic(self.cfg, particle_fdim, self.ddlp_model.action_dim, n_critics=1,
                                 out_dim=max(cfg.num_bins, 1), background_fdim=background_fdim)
        self._pi = EITActor(self.cfg, particle_fdim, self.ddlp_model.action_dim, background_fdim=background_fdim)
        self._Qs = EITCritic(self.cfg, particle_fdim, self.ddlp_model.action_dim, n_critics=cfg.num_q,
                             out_dim=max(cfg.num_bins, 1), background_fdim=background_fdim)
        self.apply(init.weight_init)
        init.zero_([self._reward.q_networks[0].output_mlp[-1].weight] + [q.output_mlp[-1].weight for q in self._Qs.q_networks])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

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
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
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
        raise NotImplementedError('Multitasking is not implemented for ddlp-based observations')

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

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z_fg = z['fg']
        z_kp = z_fg[..., :2]
        z_scale = z_fg[..., 2:4]
        z_depth = z_fg[..., 3:4]
        z_features = z_fg[..., 5:-1]
        z_obj_on = z_fg[..., -1:]
        z_bg = z['bg']

        dyn_out = self.ddlp_model.dyn_module(z_kp, z_scale, z_obj_on, z_depth, z_features, z_bg, a)
        dyn_obj_on_beta_dist = torch.distributions.Beta(dyn_out['obj_on_a'], dyn_out['obj_on_b'])
        dyn_obj_on = dyn_obj_on_beta_dist.mean
        next_z_fg = self.ddlp_model.get_dlp_rep(dyn_out['mu'], dyn_out['mu_scale'], dyn_out['mu_depth'], dyn_out['mu_features'], dyn_obj_on.unsqueeze(-1))

        return {'fg': next_z_fg, 'bg': dyn_out['mu_bg_features']}

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        x = z['fg'][:, -1]
        action = a[:, -1]
        if self.cfg.multitask:
            x = self.task_emb(x, task)
        return self._reward(x, action, bg=z['bg'][:, -1])[0]

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        x = z['fg'][:, -1]
        if self.cfg.multitask:
            x = self.task_emb(x, task)

        # Gaussian policy prior
        mu, log_std = self._pi(x, bg=z['bg'][:, -1]).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, **kwargs):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        z_fg = z['fg']
        if len(a.size()) == len(z_fg.size()) - 1:
            action = a[:, -1]
        elif len(a.size()) == len(z_fg.size()) - 2:
            # action without timestep_horizon dimension
            action = a
        else:
            raise ValueError(f'Unexpected action shape: {a.size()}. State shape: {z.size()}')

        x = z_fg[:, -1]
        if self.cfg.multitask:
            x = self.task_emb(x, task)

        Qs = self._target_Qs if target else self._Qs
        out = torch.stack(Qs(x, action, bg=z['bg'][:, -1]))

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2


class DDLPGNNWorldModel(nn.Module):
    def __init__(self, cfg, ddlp_model: ObjectDynamicsDLP):
        super().__init__()
        self.cfg = cfg
        self.ddlp_model = ddlp_model
        assert not cfg.multitask, f'Multitasking is not implemented for slot-based observations.'

        self._encoder = layers.enc(cfg)
        self._dynamics = nn.Identity()
        slot_dim = self.ddlp_model.get_dlp_features_dim() * self.ddlp_model.timestep_horizon
        n_slots = self.ddlp_model.n_kp_enc
        if self.cfg.eit_use_background:
            n_slots += 1
            self.background_projection = mlp(
                self.ddlp_model.get_dlp_background_dim() * self.ddlp_model.timestep_horizon, [], slot_dim
            )

        action_dim = self.cfg.action_dim * self.ddlp_model.timestep_horizon
        self._reward = OCRewardModel(self.cfg, slot_dim=slot_dim, n_slots=n_slots, action_dim=action_dim)
        self._pi = OCPolicy(self.cfg, slot_dim=slot_dim, n_slots=n_slots)
        self._Qs = nn.ModuleList(
            [OCRewardModel(self.cfg, slot_dim=slot_dim, n_slots=n_slots, action_dim=action_dim) for _ in range(cfg.num_q)]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward.mlp.weight] + [q.mlp.weight for q in self._Qs])
        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)
        self.log_std_min = torch.tensor(cfg.log_std_min)
        self.log_std_dif = torch.tensor(cfg.log_std_max) - self.log_std_min

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
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
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
        raise NotImplementedError('Multitasking is not implemented for ddlp-based observations')

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

    def next(self, z, a, task):
        """
        Predicts the next latent state given the current latent state and action.
        """
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z_fg = z['fg']
        z_kp = z_fg[..., :2]
        z_scale = z_fg[..., 2:4]
        z_depth = z_fg[..., 3:4]
        z_features = z_fg[..., 5:-1]
        z_obj_on = z_fg[..., -1:]
        z_bg = z['bg']

        dyn_out = self.ddlp_model.dyn_module(z_kp, z_scale, z_obj_on, z_depth, z_features, z_bg, a)
        dyn_obj_on_beta_dist = torch.distributions.Beta(dyn_out['obj_on_a'], dyn_out['obj_on_b'])
        dyn_obj_on = dyn_obj_on_beta_dist.mean
        next_z_fg = self.ddlp_model.get_dlp_rep(dyn_out['mu'], dyn_out['mu_scale'], dyn_out['mu_depth'], dyn_out['mu_features'], dyn_obj_on.unsqueeze(-1))

        return {'fg': next_z_fg, 'bg': dyn_out['mu_bg_features']}

    def reward(self, z, a, task):
        """
        Predicts instantaneous (single-step) reward.
        """
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        x = z['fg'].permute((0, 2, 1, 3)).flatten(start_dim=-2)
        if self.cfg.eit_use_background:
            self.background_projection(z['bg'].flatten(start_dim=-2))
            x = torch.cat([x, self.background_projection(z['bg'].flatten(start_dim=-2)).unsqueeze(1)], dim=1)

        action = a.flatten(start_dim=-2)
        if self.cfg.multitask:
            x = self.task_emb(x, task)
        return self._reward(x, action)

    def pi(self, z, task):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        x = z['fg'].permute((0, 2, 1, 3)).flatten(start_dim=-2)
        if self.cfg.eit_use_background:
            self.background_projection(z['bg'].flatten(start_dim=-2))
            x = torch.cat([x, self.background_projection(z['bg'].flatten(start_dim=-2)).unsqueeze(1)], dim=1)

        if self.cfg.multitask:
            x = self.task_emb(x, task)

        # x.shape -> batch_size, n_particles, features_dim * timestep_horizon
        # Gaussian policy prior
        mu, log_std = self._pi(x).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mu)

        if self.cfg.multitask:  # Mask out unused action dimensions
            mu = mu * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:  # No masking
            action_dims = None

        log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = math.squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def Q(self, z, a, task, return_type='min', target=False, prev_actions=None):
        """
        Predict state-action value.
        `return_type` can be one of [`min`, `avg`, `all`]:
            - `min`: return the minimum of two randomly subsampled Q-values.
            - `avg`: return the average of two randomly subsampled Q-values.
            - `all`: return all Q-values.
        `target` specifies whether to use the target Q-networks or not.
        """
        assert return_type in {'min', 'avg', 'all'}
        # z.shape -> batch_size, timestep_horizon, n_particles, features_dim
        x = z['fg'].permute((0, 2, 1, 3)).flatten(start_dim=-2)
        if self.cfg.eit_use_background:
            self.background_projection(z['bg'].flatten(start_dim=-2))
            x = torch.cat([x, self.background_projection(z['bg'].flatten(start_dim=-2)).unsqueeze(1)], dim=1)

        # x.shape -> batch_size, n_particles, timestep_horizon * features_dim
        if self.cfg.multitask:
            x = self.task_emb(x, task)

        if prev_actions is not None:
            prev_actions[:, :-1] = prev_actions[:, 1:]
            prev_actions[:, -1] = a
            a = prev_actions

        action = a.flatten(start_dim=-2)
        Qs = self._target_Qs if target else self._Qs
        out = torch.stack([q(x, action) for q in Qs])

        if return_type == 'all':
            return out

        Q1, Q2 = out[np.random.choice(self.cfg.num_q, 2, replace=False)]
        Q1, Q2 = math.two_hot_inv(Q1, self.cfg), math.two_hot_inv(Q2, self.cfg)
        return torch.min(Q1, Q2) if return_type == 'min' else (Q1 + Q2) / 2
