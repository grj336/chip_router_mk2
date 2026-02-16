# Config.py
# Holds hyper parameters for the chip placement environment
from dataclasses import dataclass

import torch


@dataclass
class EnvironmentConfig:
    """Configuration for the chip placement environment."""

    grid_height: int = 32
    grid_width: int = 32
    num_components_min: int = 10
    num_components_max: int = 16
    edge_probability: float = 0.3
    wirelength_weight: float = -1.0
    illegal_placement_penalty: float = -100.0


@dataclass
class GNNConfig:
    """GNN encoder settings"""

    hidden_dim: int = 64
    num_layers: int = 3
    node_feature_dim: int = 8


@dataclass
class CNNConfig:
    """CNN encoder settings"""

    hidden_channels: tuple[int, ...] = (32, 64, 64)
    kernel_sizes: tuple[int, ...] = (3, 3, 3)


@dataclass
class ExperimentConfig:
    """Main training settings"""

    env: EnvironmentConfig = None
    gnn: GNNConfig = None
    cnn: CNNConfig = None
    seed: int = 336
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.env is None:
            self.env = EnvironmentConfig()
        if self.gnn is None:
            self.gnn = GNNConfig()
        if self.cnn is None:
            self.cnn = CNNConfig()


DEFAULT_CONFIG = ExperimentConfig()
