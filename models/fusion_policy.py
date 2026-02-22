import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .cnn_encoder import CNNEncoder
from .gnn_encoder import GNNEncoder


class ChipPlacementFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for chip placement.

    Architecture:
        1. GNN processes netlist -> node embeddings
        2. CNN processes layout -> feature maps
        3. Broadcast node embedding spatially
        4. Concatenate with CNN features
        5. Apply fusion conv layers
        6. Output: spatial feature for actor + flattened for critic
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_config: dict[str, object],
        cnn_config: dict[str, object],
        fusion_config: dict[str, object],
    ):
        """Init fusion feature extractor"""
        # Extract shapes
        grid_shape = observation_space["grid"].shape  # (C, H, W)
        self.grid_channels = grid_shape[0]
        self.grid_height = grid_shape[1]
        self.grid_width = grid_shape[2]

        node_feat_shape = observation_space["node_features"].shape  # (N, F)
        self.node_feat_dim = node_feat_shape[1]

        # Build GNN encoder
        self.gnn_encoder = GNNEncoder(
            node_feature_dim=self.node_feat_dim,
            hidden_dim=gnn_config["hidden_dim"],
            num_layers=gnn_config["num_layers"],
            gnn_type=gnn_config.get("gnn_type", "GraphSAGE"),
            aggr=gnn_config.get("aggr", "mean"),
            dropout=gnn_config.get("dropout", 0.0),
            batch_norm=gnn_config.get("batch_norm", True),
        )

        # Build CNN encoder
        self.cnn_encoder = CNNEncoder(
            in_channels=self.grid_channels,
            hidden_channels=cnn_config["hidden_channels"],
            kernel_sizes=cnn_config["kernel_sizes"],
            dropout=cnn_config.get("dropout", 0.0),
            batch_norm=cnn_config.get("batch_norm", True),
        )

        # Fusion layers
        gnn_dim = self.gnn_encoder.get_output_dim()
        cnn_dim = self.cnn_encoder.get_output_channels()
        fusion_in_channels = gnn_dim + cnn_dim

        fusion_channels = fusion_config["fusion_channels"]
        fusion_kernel_sizes = fusion_config["fusion_kernel_sizes"]

        self.fusion_convs = nn.ModuleList()
        channels = [fusion_in_channels] + list(fusion_channels)

        for i in range(len(fusion_channels)):
            kernel_size = fusion_kernel_sizes[i]
            padding = kernel_size // 2

            self.fusion_convs.append(
                nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

        self.fusion_activation = nn.ReLU()

        # Feature dim for value head (flattened)
        self._features_dim = fusion_channels[-1] * self.grid_height * self.grid_width

        # Initialise base class
        super().__init__(observation_space, features_dim=self._features_dim)

        self.spatial_features: torch.Tensor | None = None

        pass
