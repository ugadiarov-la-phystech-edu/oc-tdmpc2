import gymnasium as gym
from envs.wrappers.time_limit import GymnasiumTimeLimit
from typing import Tuple

import survivalenv

import gymnasium as gym
from PIL import Image
import numpy as np
from typing import Tuple
from typing import Tuple, Optional


class GymnasiumPixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        # self.resize = torchvision.transforms.Resize(self.image_size)

    def _process_images(self, obs):
        for k, v in obs.items():
            if "image" in k:
                image = Image.fromarray(v.astype("uint8"), "RGB")
                image = image.resize(self.image_size)
                obs[k] = np.array(image)
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_images(obs)
        return obs, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset()
        obs = self._process_images(obs)
        return obs, info


class GymnasiumActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int = 1) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        accumulated_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            accumulated_reward += reward
            if terminated is True or truncated is True:
                break
        return obs, accumulated_reward, terminated, truncated, info


class GymnasiumTimeLimit(gym.Wrapper):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_episode_steps:
            info["TimeLimit.truncated"] = truncated
            truncated = True
        return obs, reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        self._elapsed_steps = 0
        return self.env.reset()


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    # print("<<<<<<<<<<<<<<<")
    env = gym.make(name, render_cameras=True)
    # print(">>>>>>>>>>>>>>>")
    env = GymnasiumActionRepeat(env, action_repeat)
    env = GymnasiumTimeLimit(env, max_episode_steps)
    env = GymnasiumPixels(env, image_size)
    # print("************************************************************")
    return env
