import json
from copy import deepcopy
import warnings

import gym
import isaacgym
import torch

from envs.wrappers.ddlp import DDLPExtractorWrapper
from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.slots import SlotExtractorWrapper
from envs.wrappers.tensor import TensorWrapper
from ocr.tools import Dinosaur, SlotExtractor


def missing_dependencies(task):
    raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')


try:
    from envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies
try:
    from envs.maniskill3 import make_env as make_maniskill3_env
except:
    make_maniskill3_env = missing_dependencies
try:
    from envs.robosuite_env import make_env as make_robosuite_env
except:
    make_robosuite_env = missing_dependencies
try:
    from envs.isaac.isaac_env_wrappers import make_env as make_isaac_env
except:
    make_isaac_env = missing_dependencies

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print('Creating multi-task environment with tasks:', cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError('Unknown task:', task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg, **kwargs):
    """
    Make an environment for TD-MPC2 experiments.
    """
    gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        env = None
        for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_maniskill3_env,
                   make_robosuite_env, make_isaac_env]:
            try:
                env = fn(cfg)
            except ValueError:
                pass
        if env is None:
            raise ValueError(
                f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')

    obs_type = cfg.get('obs', 'state')
    if obs_type == 'rgb':
        env = PixelWrapper(cfg, env, num_frames=cfg.num_frames, render_size=cfg.obs_size)
    elif obs_type == 'slots':
        dinosaur = Dinosaur(cfg.dino_model_name, cfg.n_slots, cfg.slot_dim, cfg.input_feature_dim, cfg.num_patches,
                            cfg.features)
        state_dict = torch.load(cfg.slot_extractor_checkpoint_path)['state_dict']
        state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}
        dinosaur.load_state_dict(state_dict)
        dinosaur = dinosaur.requires_grad_(False)
        dinosaur = dinosaur.eval()
        slot_extractor = SlotExtractor(model=dinosaur, device=cfg.slot_extractor_device)
        env = SlotExtractorWrapper(cfg, env, slot_extractor)
    elif obs_type == 'ddlp':
        ddlp = kwargs['extractor']
        assert ddlp.action_dim == env.action_space.shape[0]
        config_path = cfg.ddlp_config_path
        with open(config_path, 'r') as file_obj:
            config = json.load(file_obj)

        env = DDLPExtractorWrapper(env, ddlp, device='cuda', num_static_frames=config['num_static_frames'],
                                   train_enc_prior=config['train_enc_prior'])

    if not cfg.multitask:
        env = TensorWrapper(env)

    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except:  # Box
        cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    return env
