from typing import Callable, Tuple

import gym
import numpy as np
import torch
from gym.core import ActType, ObsType


class TorchTransformsWrapper(gym.Wrapper):
    def __init__(self, env, transforms: Callable, cuda: bool = False):
        super().__init__(env)
        self.transforms = transforms
        self.use_cuda = cuda

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs = self.env.reset(**kwargs)
        return self.observation(obs)

    def observation(self, observation):
        obs = self.transforms(observation)
        if self.use_cuda:
            obs = obs.cuda()
        return obs


class CompassWrapper(gym.Wrapper):
    def __init__(self, env, model, has_info: bool = False):
        super().__init__(env)
        self.model = model.eval()
        self.has_info = has_info
        self.memory = None
        shape = (self.model.num_slots, self.model.slots_dim)
        self.observation_space = gym.spaces.Box(
            low=np.full(shape, fill_value=-np.inf, dtype=np.float32),
            high=np.full(shape, fill_value=np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs = self.env.reset(**kwargs)
        if self.has_info:
            obs, info = obs

        obs = obs.unsqueeze(0).expand(self.model.max_timestep, -1, -1, -1)
        next_obs, _ = self.model.get_start_slots(obs)
        next_obs = next_obs.squeeze(0)
        self.memory = next_obs[1:]

        result = next_obs[-1]
        if self.has_info:
            result = (result, info)

        return result

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        next_obs, _ = self.model.get_next_slot(obs, self.memory), info
        self.memory = torch.cat((self.memory[1:], next_obs))
        return next_obs.squeeze(0), reward, done,  info
