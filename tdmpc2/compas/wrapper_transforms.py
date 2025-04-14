from typing import Callable

import gym
import torch


class TorchTransformsWrapper(gym.Wrapper):
    def __init__(self, env, transforms: Callable, cuda: bool = False):
        super().__init__(env)
        self.transforms = transforms
        self.use_cuda = cuda

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        if len(observation.shape) == 4:
            obs = [self.transforms(obs) for obs in observation]
            obs = torch.stack(obs, dim=0)
        else:
            obs = self.transforms(observation)
        if self.use_cuda:
            obs = obs.cuda()
        return obs
