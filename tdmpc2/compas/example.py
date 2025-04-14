from pathlib import Path

import numpy as np
import torch

from compas.config_pushing import PushingTaskConfig, Difficulty, CausalWorldEnvironmentConfig, CausalWorldConfig
from compas.cw_env import MyCausalWorld
from compas.cw_pushing_task_generator import PushingTaskGenerator
from compas.modules import DinoEncoderConfig, CompasSlotsExtractorAdapterConfig, CompasExtractorAdapter
from compas.tensor import TensorWrapper
from compas.transforms import DefaultObsTransforms
from compas.wrapper_compas import CompassWrapper
from compas.wrapper_cw import CWMaskAndImageWrapper, AutoInterventionWrapper, CausalWorldTDMPCWrapper
from compas.wrapper_transforms import TorchTransformsWrapper

image_size = 224
patch_size = 8

visual_resolution = image_size // patch_size
num_patches = visual_resolution ** 2

train_batch_size = 8
val_batch_size = 8
num_workers = 1
grad_accum = 4

run_name = "compas_causal_world_push"

num_slots = 8
slots_size = 128
video_len = 4

max_timestep = video_len

act_type = 'continuous'
num_steps = 100

min_warmup_steps = 20
max_warmup_steps = 80

seed = 1


vit_config = DinoEncoderConfig(
    version=1,
    model_size="small",
    resolution=image_size,
    patch_size=patch_size,
    frozen=True,
)

feat_dim = vit_config.resolve_feat_dim()

slots_extractor_config = CompasSlotsExtractorAdapterConfig(
    weights_path=Path("/media/elfray/hdd_ext4/projects/compas_2/checkpoint/cw/extractor.ckpt"),
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
    difficulty=Difficulty.HARD,
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



nv_dict = {}
if env_config.env is not None:
    env_dict = env_config.env.model_dump()
task = PushingTaskGenerator(**env_config.task.model_dump())
env = MyCausalWorld(task=task, seed=seed, **env_dict)
env = CWMaskAndImageWrapper(AutoInterventionWrapper(env))
env = CausalWorldTDMPCWrapper(env)

env = TorchTransformsWrapper(env, transforms, cuda=True)
env = TensorWrapper(env)
compas = CompasExtractorAdapter(
    **slots_extractor_config.shallow_dump(),
).eval().cuda()

env = CompassWrapper(env, compas, has_info=True)

obs = env.reset()

print(obs.shape)

done = False
i = 0
while not done:
    _, _, done, _ = env.step(env.action_space.sample())
    i += 1

print(i)

# obs, *_ = env.step(torch.tensor(env.sample_rand_action()))

print(obs.shape)
