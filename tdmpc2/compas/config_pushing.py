from enum import Enum

import numpy as np

from compas.base_config import BaseConfig


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class PushingTaskConfig(BaseConfig):
    variables_space: str = 'space_a_b'
    fractional_reward_weight: float = 1.
    dense_reward_weights: object = np.array([750, 250, 100])
    activate_sparse_reward: bool = False
    tool_block_mass: float = 0.02
    joint_positions: object = None
    difficulty: Enum = Difficulty.EASY
    tool_block_color: str = 'yellow'
    tool_block_position: object = np.array([0, -0.08, 0.0325])
    tool_block_orientation: object = np.array([0, 0, 0, 1])
    goal_block_color: str = 'pink'
    goal_block_position: object = np.array([0, 0.08, 0.0325])
    goal_block_orientation: object = np.array([0, 0, 0, 1])


class CausalWorldEnvironmentConfig(BaseConfig):
    skip_frame: int = 10
    enable_visualization: bool = False
    action_mode: str = "joint_positions"
    observation_mode: str = "structured",
    normalize_actions: bool = True,
    normalize_observations: bool = True,
    max_episode_length: object = None,
    camera_indicies: object = np.array([0, 1, 2]),


class CausalWorldConfig(BaseConfig):
    task: object = None
    env: object = None