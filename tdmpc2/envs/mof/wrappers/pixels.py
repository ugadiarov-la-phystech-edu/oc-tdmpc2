import gym
from multi_object_fetch.env import MultiObjectFetchEnv
from PIL import Image
import numpy as np
from typing import Tuple, Union


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Union[Tuple[int, int], int], normalize_image: bool = False) -> None:
        super().__init__(env)
        self.normalize_image = normalize_image
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        shape = (3,) + self.image_size if self.normalize_image else self.image_size + (3,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=float)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if isinstance(self.env.unwrapped, MultiObjectFetchEnv):
            image = self.env.render(mode='rgb_array', size=self.image_size)
        else:
            image = Image.fromarray(self.env.render(mode='rgb_array'))
            image = np.array(image.resize(self.image_size))

        if self.normalize_image:
            return np.moveaxis(image, -1, 0) / 255.0

        return image.copy()

    def render(self, mode='human', **kwargs):
        size = self.image_size
        if 'width' and 'height' in kwargs:
            size = (kwargs['width'], kwargs['height'])

        return self.env.render(mode='rgb_array', size=size)
