from pathlib import Path

import numpy as np
import torch

from compas.config_pushing import PushingTaskConfig, Difficulty, CausalWorldEnvironmentConfig, CausalWorldConfig
from compas.cw_env import MyCausalWorld
from compas.cw_pushing_task_generator import PushingTaskGenerator
from compas.modules import DinoEncoderConfig, CompasSlotsExtractorAdapterConfig, CompasExtractorAdapter
from compas.tensor import TensorWrapper
from compas.transforms import DefaultObsTransforms
from compas.transition_compas import CompasDynamicsModel
from compas.wrapper_compas import CompassWrapper
from compas.wrapper_cw import CWMaskAndImageWrapper, AutoInterventionWrapper, CausalWorldTDMPCWrapper
from compas.wrapper_transforms import TorchTransformsWrapper


def make_cw_compas_env(cfg):
    env_name = cfg['task']
    if env_name != 'cw_push':
        raise ValueError(f'Unexpected task for Robosuite environment: {env_name}')

    difficulty = cfg.difficulty
    seed = cfg.seed
    device = cfg.slot_extractor_device
    compas_extractor_path = cfg.slot_extractor_checkpoint_path

    if difficulty == 'easy':
        difficulty = Difficulty.EASY
    elif difficulty == 'medium':
        difficulty = Difficulty.MEDIUM
    elif difficulty == 'hard':
        difficulty = Difficulty.HARD
    else:
        assert False

    image_size = 224
    patch_size = 8

    visual_resolution = image_size // patch_size
    num_patches = visual_resolution ** 2

    num_slots = 8
    slots_size = 128
    video_len = 4

    max_timestep = video_len

    vit_config = DinoEncoderConfig(
        version=1,
        model_size="small",
        resolution=image_size,
        patch_size=patch_size,
        frozen=True,
    )

    feat_dim = vit_config.resolve_feat_dim()

    slots_extractor_config = CompasSlotsExtractorAdapterConfig(
        weights_path=Path(compas_extractor_path),
        encoder_config=vit_config,
        num_slots=num_slots,
        slots_dim=slots_size,
        num_layers=4,
        max_timestep=max_timestep,
        feat_dim=feat_dim,
        num_patches=num_patches)

    task_config = PushingTaskConfig(
        variables_space='space_a_b',
        fractional_reward_weight=1.,
        dense_reward_weights=np.array([750, 250, 100]),
        activate_sparse_reward=True,
        tool_block_mass=0.02,
        joint_positions=None,
        difficulty=difficulty,
        tool_block_color='yellow',
        tool_block_position=np.array([0, -0.08, 0.0325]),
        tool_block_orientation=np.array([0, 0, 0, 1]),
        goal_block_color='pink',
        goal_block_position=np.array([0, 0.08, 0.0325]),
        goal_block_orientation=np.array([0, 0, 0, 1]),
    )

    cw_config = CausalWorldEnvironmentConfig(
        skip_frame=10,
        enable_visualization=False,
        action_mode="joint_positions",
        observation_mode="structured",
        normalize_actions=True,
        normalize_observations=True,
        max_episode_length=None,
        camera_indicies=np.array([0, 1, 2])
    )

    env_config = CausalWorldConfig(
        task=task_config,
        env=cw_config,
    )

    transforms = DefaultObsTransforms(
        resolution=image_size,
    )

    if env_config.env is not None:
        env_dict = env_config.env.model_dump()
    task = PushingTaskGenerator(**env_config.task.model_dump())
    env_dict['max_episode_length'] = cfg.time_limit
    env = MyCausalWorld(task=task, seed=seed, **env_dict)
    env = CWMaskAndImageWrapper(AutoInterventionWrapper(env))
    env = CausalWorldTDMPCWrapper(env)

    env = TorchTransformsWrapper(env, transforms, cuda=True)
    env = TensorWrapper(env)
    compas = CompasExtractorAdapter(
        **slots_extractor_config.shallow_dump(),
    ).eval().to(device)
    compas.requires_grad_(False)
    env.max_episode_steps = env.unwrapped._max_episode_length

    return CompassWrapper(env, compas, has_info=True)


def create_transition_model(device, path):
    num_slots = 8
    slots_size = 128
    tokens_dim = slots_size * 2
    actions_dim = 9
    projected_actions_dim = tokens_dim
    max_timestep = 4
    num_layers = 4
    parallel = True

    model = CompasDynamicsModel(
        num_slots=num_slots,
        slots_dim=slots_size,
        tokens_dim=tokens_dim,
        max_timestep=max_timestep,
        actions_dim=actions_dim,
        projected_actions_dim=projected_actions_dim,
        num_layers=num_layers,
        parallel=parallel,
    )

    model.load_state_from_method(torch.load(path)['state_dict'])
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    return model
