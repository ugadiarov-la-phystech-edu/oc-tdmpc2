from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()
        self._tds = None
        if self.cfg.get('checkpoint', None):
            print(f'Loading checkpoint: {self.cfg.checkpoint}')
            state_dict = torch.load(self.cfg.checkpoint)
            self.agent.load(state_dict)
            self._step = state_dict['step']
            self._ep_idx = state_dict['episode']
            self.buffer.num_eps = state_dict['episode']
            self._start_time = time() - state_dict['total_time']

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        total_time = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=total_time,
            fps= self._step / total_time
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        total_time = 0
        total_steps = 0
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            start_time = time()
            while not done:
                previous_actions = None
                if self.cfg.obs == 'ddlp':
                    previous_actions = torch.from_numpy(self.env.get_actions()).to(obs['fg'].device)

                action = self.agent.act(obs, t0=t == 0, eval_mode=True, prev_actions=previous_actions)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            total_time += time() - start_time
            total_steps += t
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            fps=total_steps / total_time,
        )

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict({k: v.unsqueeze(0) for k, v in obs.items()}, batch_size=(), device='cpu')
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            if self.cfg.obs == 'ddlp':
                action = torch.full(self.env.get_actions().shape, float('nan'), dtype=torch.float32)
            else:
                action = torch.full_like(self.env.rand_act(), float('nan'))

        if reward is None:
            reward = torch.tensor(float('nan'))
        td = TensorDict(dict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
        ), batch_size=(1,))
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, self.cfg.eval_freq > 0
        next_save_step = self._step + self.cfg.save_every
        while self._step <= self.cfg.steps:

            # Evaluate agent periodically
            if self.cfg.eval_freq > 0 and self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    eval_next = False

                if self._tds is not None:
                    train_metrics.update(
                        episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
                        episode_success=info['success'],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]
                if not self.buffer.is_initialized():
                    self.buffer.init(torch.cat(self._tds))

            # Collect experience
            if self._step > self.cfg.seed_steps:
                if self.cfg.obs == 'ddlp':
                    action = self.agent.act(obs, t0=len(self._tds) == 1, prev_actions=torch.from_numpy(self.env.get_actions()))
                else:
                    action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, info = self.env.step(action)
            if self.cfg.obs == 'ddlp':
                buffer_action = torch.from_numpy(self.env.get_actions())
            else:
                buffer_action = action

            self._tds.append(self.to_td(obs, buffer_action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print('Pretraining agent on seed data...')
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            if self._step > next_save_step:
                next_save_step += self.cfg.save_every
                self.logger.save_agent(self.agent, statistics=self.common_metrics(), identifier='checkpoint', buffer=self.buffer)

            self._step += 1

        self.logger.finish(self.agent)
