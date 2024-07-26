import gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit

import gymnasium
import mani_skill.envs

MANISKILL_TASKS = {
    'push-cube': dict(
        env='PushCube-v1',
        control_mode='pd_joint_delta_pos',
    ),
}


class ManiSkillWrapper(gym.Wrapper):
    def __init__(self, env, cfg, frame_skip=1):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.cfg.obs_size, self.cfg.obs_size, 3),
                                                dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape, self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )
        self.env.reset(seed=cfg.seed)
        self.frame_skip = frame_skip
        self.last_observation = None

    @staticmethod
    def _unravel(step_result):
        unravel_result = [step_result[0]['sensor_data']['base_camera']['rgb'][0]]
        unravel_result += [x[0] if hasattr(x, '__len__') else x for x in step_result[1:-1]]
        info = {key: value[0] if hasattr(value, '__len__') else value for key, value in step_result[-1].items()}
        unravel_result.append(info)

        return unravel_result

    def reset(self):
        self.last_observation = self._unravel(self.env.reset())[0].numpy()
        return self.last_observation.copy()

    def step(self, action):
        reward = 0
        for _ in range(self.frame_skip):
            obs, r, terminated, truncated, info = self._unravel(self.env.step(action))
            reward += r
            if terminated or truncated:
                break

        self.last_observation = obs.numpy()
        return self.last_observation.copy(), reward, terminated, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return self.last_observation.copy()


def make_env(cfg):
    """
    Make ManiSkill3 environment.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError('Unknown task:', cfg.task)
    assert cfg.obs in ('rgb', 'slots'), 'This task supports only image-based and slot-based observations.'
    task_cfg = MANISKILL_TASKS[cfg.task]
    env = gymnasium.make(
        task_cfg['env'],
        obs_mode='rgbd',
        control_mode=task_cfg['control_mode'],
        render_mode='rgb_array',
        sensor_configs=dict(width=cfg.obs_size, height=cfg.obs_size),
    )
    # Unwrap TimeLimit wrapper
    env = env.env
    env = ManiSkillWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=100)
    env.max_episode_steps = env._max_episode_steps
    return env
