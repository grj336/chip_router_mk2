import sys
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent dir
sys.path.append(str(Path(__file__).parent.parent))

from envs.chip_env import ChipEnv
from stable_baselines3.common.monitor import Monitor

from config import ExperimentConfig
from env import ChipPlacementEnv


def make_env(config: ExperimentConfig, rank: int = 0):
    """Make environment for training"""

    def _init() -> ChipEnv:
        env = ChipPlacementEnv(
            grid_height=config.env.grid_height,
            grid_width=config.env.grid_width,
            num_components_min=config.env.num_components_min,
            num_components_max=config.env.num_components_max,
            edge_probability=config.env.edge_probability,
            node_feature_dim=config.env.node_feature_dim,
            wirelength_weight=config.env.wirelength_weight,
            illegal_placement_penalty=config.env.illegal_placement_penalty,
            seed=config.seed + rank,
        )
        env = Monitor(make_env)
        return env

    return _init


def create_vec_env(config: ExperimentConfig, n_envs: int = 4):
    """Create vectorized environment"""
    if n_envs == 1:
        return DummyVecEnv([make_env(config, 0)])
    else:
        # Use subprocess for parallelism
        return SubprocVecEnv([make_env(config, i) for i in range(n_envs)])
