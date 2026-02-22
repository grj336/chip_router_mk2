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
        # Feature dim for value head (flattened)
        grid_shape = observation_space["grid"].shape  # (C, H, W)
        fusion_channels = fusion_config["fusion_channels"]
        features_dim = fusion_channels[-1] * grid_shape[1] * grid_shape[2]

        # Initialise base class
        super().__init__(observation_space, features_dim=features_dim)

        # Extract shapes
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
        self.spatial_features: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass - get features from observations"""

        # Unpack observations
        grid = observations["grid"]  # (B, C, H, W)
        node_features = observations["node_features"]  # (B, N, F)
        edge_index = observations["edge_index"]  # (2, E)
        current_node = observations["current_node"]  # (B,)

        batch_size = grid.shape[0]  # (B,)

        # Process spatial features with CNN encoder
        cnn_features = self.cnn_encoder(grid)  # (B, C_cnn, H, W)

        # Process graph with GNN (batched)
        gnn_embeddings_list = []

        for i in range(batch_size):
            # Extract graph data for current sample
            node_feat_i = node_features[i]  # (N_i, F)
            edge_index_i = edge_index[i]  # (2, E_i)
            current_node_i = current_node[i : i + 1]  # (1,)

            # Filter out padding edges
            valid_edges = ~((edge_index_i[0] == 0) & (edge_index_i[1] == 0))
            edge_index_i = edge_index_i[:, valid_edges]

            # Get embedding for current node
            node_embedding = self.gnn_encoder(
                node_features=node_feat_i, edge_index=edge_index_i, current_node=current_node_i
            )  # (1, hidden_dim)

            gnn_embeddings_list.append(node_embedding)

        gnn_ebeddings = torch.cat(gnn_embeddings_list, dim=0)  # (B, hidden_dim)

        # Broadcast node embedding to grid size
        # (B, hidden_dim) -> (B, hidden_dim, 1, 1) -> (B, hidden_dim, H, W)
        gnn_spatial = gnn_ebeddings.unsqueeze(-1).unsqueeze(-1)
        gnn_spatial = gnn_spatial.expand(-1, -1, self.grid_height, self.grid_width)

        # Concatenate with CNN features
        # (B, C_cnn, H, W) + (B, hidden_dim, H, W) -> (B, C_cnn + hidden_dim, H, W)
        fused = torch.cat([cnn_features, gnn_spatial], dim=1)  # (B, C_cnn + hidden_dim, H, W)

        # Apply fusion layers
        x = fused
        for conv in self.fusion_convs:
            x = conv(x)
            x = self.fusion_activation(x)

        # Store spatial features for actor head
        self.spatial_features = x  # (B, C_fusion, H, W)

        # Flatten for value head
        features = x.flatten(start_dim=1)  # (B, C_fusion * H * W)

        return features

    def get_spatial_features(self) -> torch.Tensor:
        """Get spatial features for actor head"""
        assert (
            self.spatial_features is not None
        ), "Spatial features not computed. Run forward pass first."
        return self.spatial_features
