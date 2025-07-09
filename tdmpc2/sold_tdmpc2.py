import numpy as np
import torch
import torch.nn.functional as F
from tensordict import set_lazy_legacy, TensorDict

from common import math
from common.scale import RunningScale
from common.sold_world_model import SoldWorldModel
from common.world_model import WorldModel, OCWorldModel, DDLPWorldModel, DDLPGNNWorldModel


class SoldTDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg, ddlp_model=None):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device)
        if self.cfg.obs == 'slots':
            if self.cfg.world_model_type == 'sold':
                self.model = SoldWorldModel(cfg).to(self.device)
            else:
                self.model = OCWorldModel(cfg).to(self.device)
        elif self.cfg.obs == 'ddlp':
            if self.cfg.world_model_type == 'eit':
                self.model = DDLPWorldModel(cfg, ddlp_model=ddlp_model).to(self.device)
            elif self.cfg.world_model_type == 'gnn':
                self.model = DDLPGNNWorldModel(cfg, ddlp_model=ddlp_model).to(self.device)
            else:
                raise ValueError(f'Unexpected world model type: {self.cfg.world_model_type}')
        else:
            self.model = WorldModel(cfg).to(self.device)

        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)  # Heuristic for large action spaces
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.cfg.device
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length / self.cfg.discount_denom
        return min(max((frac - 1) / (frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, statistics, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            statistics (dict): Statistics (step, metrics, etc.)
            fp (str): Filepath to save state dict to.
        """
        checkpoint = {"model": self.model.state_dict(), "optim": self.optim.state_dict(),
                      "pi_optim": self.pi_optim.state_dict(), "scale": self.scale.state_dict()}
        checkpoint.update(statistics)
        torch.save(checkpoint, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])
        self.pi_optim.load_state_dict(state_dict["pi_optim"])
        self.scale.load_state_dict(state_dict["scale"])

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None, prev_actions=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        if self.cfg.obs == 'ddlp':
            obs = {k: v.to(self.device, non_blocking=True).unsqueeze(0) for k, v in obs.items()}
        else:
            obs = torch.stack(list(obs)).to(self.device, non_blocking=True).unsqueeze(0) # (1, obs.shape)

        if t0:
            prev_actions = torch.empty(size=(obs.size()[0], 0, self.cfg.action_dim), device=self.device, dtype=torch.float32)
        else:
            prev_actions = torch.as_tensor(np.stack(prev_actions), device=self.cfg.device).unsqueeze(0)

        if task is not None:
            task = torch.tensor([task], device=self.device)
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task, prev_actions=prev_actions)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        discount = torch.pow(torch.full(size=(self.cfg.horizon + 1,), fill_value=self.discount, device=self.device),
                             torch.arange(self.cfg.horizon + 1, device=self.device))
        actions = actions.swapaxes(0, 1)
        all_z = torch.cat([z, self.model.next(z, actions, task)], dim=1)
        rewards = math.two_hot_inv(self.model.reward(all_z[:, :-1], actions, task, start=z.shape[1] - 1), self.cfg)
        G = torch.sum(rewards.squeeze(2) * discount[:-1].unsqueeze(0), dim=-1, keepdim=True)
        all_actions = torch.cat([actions, self.model.pi(all_z, task)[1].unsqueeze(1)], dim=1)

        return G + discount[-1] * self.model.Q(all_z, all_actions, task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None, prev_actions=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            z (torch.Tensor): Latent state from which to plan.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        # Sample policy trajectories
        num_context = z.size()[1]
        num_action_context = prev_actions.size()[1]
        assert num_action_context == num_context - 1, f'Expected: {num_context - 1}. Actual: {num_action_context}.'
        horizon = self.cfg.horizon + num_action_context

        if self.cfg.num_pi_trajs > 0:
            if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
                pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.model.ddlp_model.timestep_horizon, self.cfg.action_dim, device=self.device)
                pi_actions[0, :, :-1] = prev_actions[:, 1:]
            else:
                pi_actions = torch.empty(horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
                pi_actions[:num_action_context] = prev_actions.swapaxes(0, 1)

            if self.cfg.obs == 'ddlp':
                _z = {k: torch.repeat_interleave(v, repeats=self.cfg.num_pi_trajs, dim=0) for k, v in z.items()}
            else:
                _z = torch.repeat_interleave(z, repeats=self.cfg.num_pi_trajs, dim=0)

            for t in range(num_action_context, horizon - 1):
                actions = self.model.pi(_z, task)[1] # _z: 24, emb_size, pi_actions: horizon, 24, action_dim
                if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
                    pi_actions[t, :, -1] = actions
                    pi_actions[t + 1, :, :-1] = pi_actions[t, :, 1:]
                else:
                    pi_actions[t] = actions

                _z = torch.cat([_z, self.model.next(_z, pi_actions[:t + 1].swapaxes(0, 1), task)], dim=1)
            actions = self.model.pi(_z, task)[1]
            if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
                pi_actions[-1, :, -1] = actions
            else:
                pi_actions[-1] = actions

        # Initialize state and parameters
        if self.cfg.obs == 'ddlp':
            z = {k: torch.repeat_interleave(v, repeats=self.cfg.num_samples, dim=0) for k, v in z.items()}
        else:
            z = torch.repeat_interleave(z, repeats=self.cfg.num_samples, dim=0)

        if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
            actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.model.ddlp_model.timestep_horizon, self.cfg.action_dim, device=self.device)
        else:
            actions = torch.empty(horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
            actions[:num_action_context] = prev_actions.swapaxes(0, 1)

        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions

        if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
            actions[0, self.cfg.num_pi_trajs:, :-1] = prev_actions[:, 1:]

        # Iterate MPPI
        for _ in range(self.cfg.iterations):

            # Sample actions
            sampled_actions = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                               torch.randn(self.cfg.horizon,
                                           self.cfg.num_samples - self.cfg.num_pi_trajs,
                                           self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
                actions[:, self.cfg.num_pi_trajs:, -1] = sampled_actions
                for t in range(1, actions.size()[0]):
                    actions[t, self.cfg.num_pi_trajs:, :-1] = actions[t - 1, self.cfg.num_pi_trajs:, 1:]
            else:
                actions[-self.cfg.horizon:, self.cfg.num_pi_trajs:] = sampled_actions
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            # Compute elite actions
            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[-self.cfg.horizon:, elite_idxs]
            if self.cfg.obs == 'ddlp' and self.cfg.transition_model_type == 'ddlp':
                elite_actions = elite_actions[:, :, -1]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(
                torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        # Select action
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)

    def update_pi(self, zs, task, prev_actions=None):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        batch_size, seq_len = zs.size()[:2]
        start_prediction_index = seq_len - self.cfg.horizon
        # zs = zs.flatten(end_dim=1)
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task, start=0)
        if prev_actions is not None:
            prev_actions = prev_actions.flatten(end_dim=1)

        qs = self.model.Q(zs, pis, task, return_type='avg', prev_actions=prev_actions, start=start_prediction_index)
        self.scale.update(qs[0])
        qs = self.scale(qs)
        log_pis = log_pis[:, -self.cfg.horizon:]

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon, device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(0, -1)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, z, reward, task, prev_actions=None):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            z (torch.Tensor): Latent state.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        # next_z -> batch_size, horizon, *
        # prev_actions -> batch_size, horizon, action_dim
        batch_size, seq_len = z.size()[:2]
        start_prediction_index = seq_len - self.cfg.horizon
        pi = self.model.pi(z, task, start=0)[1]
        if prev_actions is not None:
            prev_actions = prev_actions.flatten(end_dim=1)
            pi = torch.cat([prev_actions[:, 1:], pi.unsqueeze(-2)], dim=-2)

        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(z, pi, task, return_type='min', target=True, start=start_prediction_index)

    def update(self, buffer):
        """
        Main update function. Corresponds to one iteration of model learning.

        Args:
            buffer (common.buffer.Buffer): Replay buffer.

        Returns:
            dict: Dictionary of training statistics.
        """
        obs, action, reward, task = buffer.sample()
        next_action = action[1:].swapaxes(0, 1)
        reward = reward.swapaxes(0, 1)[:, -self.cfg.horizon:]

        # Compute targets
        with torch.no_grad():
            z = self.model.encode(obs, task).swapaxes(0, 1)
            prev_z = z[:, :-1]
            next_z = z[:, -self.cfg.horizon:]
            z_context = z[:, :-self.cfg.horizon]
            prev_actions = None
            if self.cfg.obs == 'ddlp' and self.cfg.world_model_type == 'eit' and self.cfg.transition_model_type == 'ddlp':
                prev_actions = next_action

            td_targets = self._td_target(z, reward, task, prev_actions=prev_actions)

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        weight = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon, device=self.device))
        consistency_loss = torch.as_tensor(0, dtype=torch.float32, device=self.device)
        if self.cfg.obs == 'ddlp':
            zs = [obs[0]]
            # Latent rollout
            for t in range(self.cfg.horizon):
                zs.append(TensorDict(self.model.next(zs[-1], next_action[t], task), batch_size=obs[0].batch_size))
                if self.cfg.transition_model_type == 'gnn':
                    consistency_loss += F.mse_loss(zs[-1]['fg'][:, -1], next_z[t]['fg'][:, -1]) * self.cfg.rho ** t

            with set_lazy_legacy(False):
                zs = torch.stack(zs, dim=0)

            # Predictions
            _zs = zs[:-1]
        else:
            # Latent rollout
            # Predictions
            _zs = self.model.next(z_context, next_action, task)
            consistency_loss = torch.sum(torch.mean((_zs - next_z) ** 2, dim=(0, 2, 3)) * weight)

        batch_size, seq_len = prev_z.size()[:2]
        start_prediction_index = seq_len - self.cfg.horizon

        # Compute losses
        qs = self.model.Q(prev_z, next_action, task, return_type='all', start=start_prediction_index)
        assert qs.shape[-2] == self.cfg.horizon
        value_loss = math.soft_ce(qs.flatten(end_dim=2), td_targets.unsqueeze(0).expand(qs.shape[0], -1, -1, -1).flatten(end_dim=2), self.cfg)
        value_loss = value_loss.reshape(*qs.shape[:3], -1)
        value_loss = torch.sum(torch.mean(value_loss, dim=(0, 1, 3)) * weight)

        reward_preds = self.model.reward(prev_z, next_action, task, start=start_prediction_index)
        assert reward_preds.shape[1] == self.cfg.horizon
        reward_loss = math.soft_ce(reward_preds.flatten(end_dim=1), reward.flatten(end_dim=1), self.cfg)
        reward_loss = torch.sum(torch.mean(reward_loss.reshape(batch_size, self.cfg.horizon, -1), dim=(0, 2)) * weight)

        consistency_loss *= (1 / self.cfg.horizon)
        reward_loss *= (1 / self.cfg.horizon)
        value_loss *= (1 / (self.cfg.horizon * self.cfg.num_q))
        total_loss = (
                self.cfg.consistency_coef * consistency_loss +
                self.cfg.reward_coef * reward_loss +
                self.cfg.value_coef * value_loss
        )

        # Update model
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        # Update policy
        prev_actions = None
        if self.cfg.obs == 'ddlp' and self.cfg.world_model_type == 'eit' and self.cfg.transition_model_type == 'ddlp':
            prev_actions = action.detach()

        pi_loss = self.update_pi(torch.cat([z_context, _zs], dim=1).detach(), task, prev_actions)

        # Update target Q-functions
        self.model.soft_update_target_Q()

        # Return training statistics
        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
        }
