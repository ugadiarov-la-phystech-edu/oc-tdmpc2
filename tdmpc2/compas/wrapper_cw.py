import numpy as np
from causal_world.envs import CausalWorld
import gym

from compas.cw_env import MyCausalWorld


class CWPixelWrapper(gym.Wrapper):
    def __init__(self, env: CausalWorld):
        super().__init__(env)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.env.render(mode="rgb_array")['image']

    def step(self, action):
        obs, rew, done, info = super().step(action)
        obs = self.env.render(mode="rgb_array")['image']
        return obs, rew, done, info


class CWMaskAndImageWrapper(gym.Wrapper):
    def __init__(self, env: CausalWorld):
        super().__init__(env)

    def reset(self, **kwargs):
        super().reset(**kwargs)
        return self.env.render(mode="rgb_array")

    def step(self, action):
        obs, rew, done, info = super().step(action)
        obs = self.env.render(mode="rgb_array")

        return obs, rew, done, info

class CausalWorldTDMPCWrapper(gym.Wrapper):
    def __init__(self, env: MyCausalWorld, frame_stack: int = 4):
        super().__init__(env)
        self.env = env
        # self.max_episode_steps = self.env.
        self.frame_stack = frame_stack

    def reset(self, **kwargs):
        obs = self.env.reset()
        obss = []
        for _ in range(self.frame_stack):
            obss.append(self.render())

        obs = np.array(obss)
        return obs, {}
    def sample_rand_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        obs, r, done, info = self.env.step(action.copy())
        obs = obs['image']
        return obs, r, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def obs_shape(self):
        return self.env.observation_space.shape

    def render(self, *args, **kwargs):
        # self.env.render()
        obs = self.env.render(mode="rgb_array")
        return obs['image']


class AutoInterventionWrapper(gym.Wrapper):
    def __init__(self, env: CausalWorld):
        super(AutoInterventionWrapper, self).__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        intervention = self.env.sample_new_goal()
        success_signal, obs = self.env.do_intervention(intervention)
        return obs
