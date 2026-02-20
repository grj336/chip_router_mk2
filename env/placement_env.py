"""
Basic Gymnasium environment
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .metrics import calculate_incremental_wire_length
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

    def reset(
        self, seed: int = None, options: dict[str, object] | None = None
    ) -> tuple[dict[str, object], dict[str, object]]:
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
            self.netlist_gen.rng = np.random.RandomState(seed)

        # Sample number of components
        self.num_components = self._np_random.randint(
            self.num_components_min, self.num_components_max + 1
        )

        # Generate netlist
        self.node_features, self.edge_index, _ = self.netlist_gen.generate_erdos_renyi_graph(
            num_nodes=self.num_components,
            edge_probability=self.edge_probability,
            node_feature_dim=self.node_feature_dim,
        )

        # Random placement order
        # TODO Improve this ordering
        self.placement_order = self._np_random.permutation(self.num_components).tolist()
        self.current_step = 0

        # Reset placement state - zero the grid and placed nodes
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.placed_positions = {}
        self.placed_nodes = []

        observation = self._get_observation()
        info = {"num_components": self.num_components, "num_edges": self.edge_index.shape[1]}

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        # TODO: Implement step logic
        row, col = int(action[0]), int(action[1])
        current_node = self.placement_order[self.current_step]

        # Check action is valid
        if self.occupancy_grid[row, col] > 0:
            # Illegal
            reward = self.illegal_placement_penalty
            terminated = True
            truncated = False
            info = {"illegal_placement": True, "reason": "occupied_position"}
            return self._get_observation(), reward, terminated, truncated, info

        # Legal
        # Place node
        self.occupancy_grid[row, col] = 1.0
        self.placed_positions[current_node] = (row, col)
        self.placed_nodes.append(current_node)

        reward = self._compute_reward(current_node, (row, col))

        self.current_step += 1

        terminated = self.current_step >= self.num_components
        truncated = False
        info = {
            "illegal_placement": False,
            "placed_node": current_node,
            "position": (row, col),
            "num_placed": len(self.placed_nodes),
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _compute_reward(self, node_idx: int, position: tuple[int, int]) -> float:
        "Compute incremental wirelength reward"
        wirelength = calculate_incremental_wire_length(
            node_idx=node_idx,
            position=position,
            placed_positions=self.placed_positions,
            edge_index=self.edge_index,
            norm="manhattan",
        )
        reward = self.wirelength_weight * wirelength
        return float(reward)

    def _get_observation(self):
        """Get current observation"""
        # Grid channels
        occupancy_channel = self.occupancy_grid.copy()
        congestion_channel = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        grid = np.stack([occupancy_channel, congestion_channel], axis=0)

        # Pad node features and edge index
        padded_node_features = np.zeros(
            (self.num_components_max, self.node_feature_dim), dtype=np.float32
        )
        padded_node_features[: self.num_components] = self.node_features.cpu().numpy()

        # Pad edge index
        max_edges = self.num_components_max * self.num_components_max
        padded_edge_index = np.zeros((2, max_edges), dtype=np.int64)
        actual_edges = self.edge_index.shape[1]
        padded_edge_index[:, :actual_edges] = self.edge_index.cpu().numpy()

        # Current node
        current_node = (
            self.placement_order[self.current_step]
            if self.current_step < self.num_components
            else 0
        )

        # Action mask
        action_mask = (1 - self.occupancy_grid).flatten()

        return {
            "grid": grid.astype(np.float32),
            "node_features": padded_node_features,
            "edge_index": padded_edge_index,
            "current_node": int(current_node),
            "action_mask": action_mask.astype(np.int8),
        }

    def action_masks(self) -> np.ndarray:
        """Get action mask for MaskablePPO."""
        return (1 - self.occupancy_grid).flatten().astype(bool)
