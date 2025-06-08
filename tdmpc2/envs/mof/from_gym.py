import gym

from envs.mof.wrappers.action_repeat import ActionRepeat
from envs.mof.wrappers.pixels import Pixels
from typing import Tuple, Union


def make_env(name: str, image_size: Union[Tuple[int, int], int], action_repeat: int, seed: int = 0):
    env = gym.make(name)
    env = ActionRepeat(env, action_repeat)
    env = Pixels(env, image_size)
    return env
