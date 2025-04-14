import collections

import numpy as np
import torch
from torch import nn
import gym

class CompassWrapper(gym.Wrapper):
    def __init__(self, env, model: nn.Module, has_info: bool = False):
        super().__init__(env)
        self.model = model.eval()
        self.has_info = has_info
        self.memory = None
        self.action_history = collections.deque(maxlen=self.model.max_timestep)
        self._observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.model.max_timestep, self.model.num_slots, self.model.slots_dim),
            dtype=np.float32)

    def reset(self, **kwargs):
        for _ in range(self.action_history.maxlen):
            self.action_history.append(np.zeros(shape=self.action_space.shape, dtype=self.action_space.dtype))

        obs = self.env.reset(**kwargs)
        info = {}
        if self.has_info:
            obs, info = obs
        next_obs, _ = self.model.get_start_slots(obs)
        next_obs = next_obs.squeeze(0)
        self.memory = next_obs[1:]
        return next_obs

    def step(self, action):
        self.action_history.append(action)
        obs, reward, done, info = self.env.step(action)
        next_obs, _ = self.model.get_next_slot(obs, self.memory), info
        next_obs = torch.cat((self.memory, next_obs))
        self.memory = next_obs[1:]
        info['action'] = np.stack(self.action_history)
        return next_obs, reward, done, info
