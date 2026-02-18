"""
Basic Gymnasium environment
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .netlist_generator import NetlistGenerator


class ChipPlacementEnv(gym.Env):
    def __init__(
        self,
        grid_height: int = 32,
        grid_width: int = 32,
        num_components_min: int = 10,
        num_components_max: int = 15,
        edge_probability: float = 0.3,
        node_feature_dim: int = 8,
        wirelength_weight: float = -1.0,
        illegal_placement_penalty: float = -100.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.num_components_min = num_components_min
        self.num_components_max = num_components_max
        self.edge_probability = edge_probability
        self.node_feature_dim = node_feature_dim
        self.wirelength_weight = wirelength_weight
        self.illegal_placement_penalty = illegal_placement_penalty

        # Netlist gen
        self.netlist_gen = NetlistGenerator(seed=seed if seed is not None else 366)

        # Action space: 2D Multidiscreet
        self.action_space = spaces.MultiDiscrete([grid_height, grid_width])

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0, high=1.0, shape=(2, grid_height, grid_width), dtype=np.float32
                ),
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_components_max, node_feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=num_components_max - 1,
                    shape=(2, num_components_max * num_components_max),
                    dtype=np.int64,
                ),
                "current_node": spaces.Discrete(num_components_max),
                "action_mask": spaces.MultiBinary(grid_height * grid_width),
            }
        )

        self.num_components: int = 0
        self.node_features: torch.Tensor | None = None
        self.edge_index: torch.Tensor | None = None
        self.placement_order: list[int] = []
        self.current_step: int = 0

        # Placement tracking
        self.occupancy_grid: np.ndarray = np.zeros((grid_height, grid_width), dtype=np.float32)
        self.placed_positions: dict[int, tuple[int, int]] = {}
        self.placed_nodes: list[int] = []

        self._np_random = np.random.RandomState(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((4, 4), dtype=np.float32)
        return self.grid, {}

    def step(self, action):
        # TODO: Implement step logic
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return self.grid, reward, terminated, truncated, info
