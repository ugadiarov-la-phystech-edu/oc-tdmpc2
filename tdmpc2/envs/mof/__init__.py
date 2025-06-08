import os

from envs.wrappers.time_limit import TimeLimit

os.environ['MUJOCO_GL'] = 'egl'  # Set MuJoCo rendering backend.
from envs.mof.from_gym import make_env as make_gym_env
# from envs.wrappers.to_tensor import ToTensor
import gym
from typing import Tuple

MOF_TASKS = {
    'ReachRed_0to4Distractors_Dense-v1', # Reach-Specific
    'ReachReddest_0to4Distractors_Dense-v1', # Reach-Specific-Relative
    'ReachOdd_2to4Distractors_Dense-v1', # Reach-Distinct
    'ReachOddGroups_4to4Distractors_Dense-v1', # Reach-Distinct-Groups
    'PickGreen_0to4Distractors_Dense-v1', # Pick-Specific
    'PickOdd_2to4Distractors_Dense-v1', # Pick-Distinct
    'PushGreen_0to4Distractors_Dense-v1', # Push-Specific
    'PushOdd_2to4Distractors_Dense-v1', # Push-Distinct
}


def make_env(cfg) -> gym.Env:
    mof_prefix = 'mof_'
    if cfg.task.startswith(mof_prefix):
        task = cfg.task[len(mof_prefix):]
        if task in MOF_TASKS:
            from envs.mof.from_mof import make_env as make_mof_env
            env = make_mof_env(task, cfg.obs_size, action_repeat=2, seed=cfg.seed)
            env = TimeLimit(env, max_episode_steps=cfg.time_limit)
            env.max_episode_steps = env._max_episode_steps
    else:
        raise ValueError('Unknown task:', cfg.task)

    # if suite == 'gym':
    #     env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed)
    # elif suite == 'mof':
    #     from envs.from_mof import make_env as make_mof_env
    #     env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed)
    # elif suite == 'dmcontrol':
    #     from envs.from_dmcontrol import make_env as make_dmcontrol_env
    #     env = make_dmcontrol_env(name, image_size, max_episode_steps, action_repeat, seed)
    # elif suite == 'survival':
    #     from envs.from_survival import make_env as make_survival_env
    #     from envs.wrappers.to_tensordict import ToTensorDict
    #     env = make_survival_env(name, image_size, max_episode_steps, action_repeat, seed)
    #     env = ToTensorDict(env)
    # else:
    #     raise ValueError(f"Unsupported environment suite: {suite}")
    # env = ToTensor(env)
    return env
