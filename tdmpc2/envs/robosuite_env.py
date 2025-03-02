import cv2
import gym
import numpy as np
import robosuite
from gym.vector.utils import spaces

from robosuite.environments.base import MujocoEnv

from pydantic_conf.base_config import BaseConfig


class RobosuiteEnvConfig(BaseConfig):
    env_name: str = "Lift"
    robots: str = "Panda"  # load a Sawyer robot and a Panda robot
    has_renderer: bool = False
    use_camera_obs: bool = True  # no on-screen rendering
    has_offscreen_renderer: bool = True  # off-screen rendering needed for image obs
    horizon: int = 200  # each episode terminates after 200 steps
    camera_names: str = 'frontview'  # use "agentview" camera for observations
    camera_heights: int  # image height
    camera_widths: int  # image width
    reward_shaping: bool = True


class RobosuiteWrapper(gym.Wrapper):
    def __init__(self, env: MujocoEnv):
        super().__init__(env)
        self.env = env
        self.low_act, self.high_act = self.env.action_spec
        self.action_space = spaces.Box(low=self.low_act, high=self.high_act)
        self.max_episode_steps = self.env.horizon
        observation_space = (env.camera_widths[0], env.camera_heights[0], 3)
        self.observation_space = gym.spaces.Box(0, 255, observation_space, dtype=np.uint8)

    @staticmethod
    def _extract_image(observation):
        return observation['frontview_image']

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._extract_image(obs)

    def sample_rand_action(self):
        return np.random.uniform(self.low_act, self.high_act)

    def step(self, action):
        obs, r, done, info = self.env.step(action.copy())
        info['success'] = self.env._check_success()
        return self._extract_image(obs), r, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def obs_shape(self):
        return self.env.observation_spec

    def render(self, *args, **kwargs):
        image = np.flipud(self._extract_image(self.env._get_observations()))
        if 'width' in kwargs and 'height' in kwargs:
            image = cv2.resize(image, dsize=(kwargs['height'], kwargs['width']), interpolation=cv2.INTER_AREA)

        return image


def make_env(cfg):
    horizon = cfg['time_limit']
    env_name = cfg['task']
    image_size = 224
    if env_name != 'Lift':
        raise ValueError(f'Unexpected task for Robosuite environment: {env_name}')

    env_config = RobosuiteEnvConfig(
        env_name="Lift",
        robots="Panda",  # load a Sawyer robot and a Panda robot
        has_renderer=False,
        use_camera_obs=True,  # no on-screen rendering
        has_offscreen_renderer=True,  # off-screen rendering needed for image obs
        horizon=horizon,  # each episode terminates after 200 steps
        camera_names='frontview',  # use "agentview" camera for observations
        camera_heights=image_size,  # image height
        camera_widths=image_size,  # image width
        reward_shaping=True
    )

    np.random.seed(cfg['seed'])
    env = robosuite.make(
        **env_config.shallow_dump())
    env = RobosuiteWrapper(env)

    return env
