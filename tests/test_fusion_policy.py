import torch
from gymnasium import spaces

from models.fusion_policy import ChipPlacementFeaturesExtractor


def test_chip_placement_features_extractor():
    """
    Test the ChipPlacementFeaturesExtractor initialization and forward pass.
    """
    batch_size = 2
    height = 16
    width = 16
    grid_channels = 4
    node_feat_dim = 8
    num_nodes = 10
    num_edges = 20

    # 1. Define observation space
    observation_space = spaces.Dict(
        {
            "grid": spaces.Box(low=0, high=1, shape=(grid_channels, height, width), dtype=float),
            "node_features": spaces.Box(
                low=-1, high=1, shape=(num_nodes, node_feat_dim), dtype=float
            ),
            "edge_index": spaces.Box(low=0, high=num_nodes - 1, shape=(2, num_edges), dtype=int),
            "current_node": spaces.Discrete(num_nodes),
        }
    )

    # 2. Define configurations
    gnn_config = {"hidden_dim": 16, "num_layers": 2, "gnn_type": "GraphSAGE", "aggr": "mean"}
    cnn_config = {"hidden_channels": (16, 32), "kernel_sizes": (3, 3)}
    fusion_config = {"fusion_channels": (32, 64), "fusion_kernel_sizes": (3, 3)}

    # 3. Create extractor
    extractor = ChipPlacementFeaturesExtractor(
        observation_space=observation_space,
        gnn_config=gnn_config,
        cnn_config=cnn_config,
        fusion_config=fusion_config,
    )

    # 4. Create dummy observations
    observations = {
        "grid": torch.randn(batch_size, grid_channels, height, width),
        "node_features": torch.randn(batch_size, num_nodes, node_feat_dim),
        "edge_index": torch.randint(0, num_nodes, (batch_size, 2, num_edges)),
        "current_node": torch.randint(0, num_nodes, (batch_size,)),
    }

    # 5. Test forward pass
    output = extractor(observations)

    # Calculate expected features dim
    # Fusion output channels is the last element of fusion_channels config
    # Spatial dimensions should be preserved by padding
    expected_output_channels = fusion_config["fusion_channels"][-1]
    expected_dim = expected_output_channels * height * width

    assert output.shape == (
        batch_size,
        expected_dim,
    ), f"Expected output shape {(batch_size, expected_dim)}, got {output.shape}"

    # 6. Test spatial features retrieval
    spatial_features = extractor.get_spatial_features()
    expected_spatial_shape = (batch_size, expected_output_channels, height, width)
    assert (
        spatial_features.shape == expected_spatial_shape
    ), f"Expected spatial shape {expected_spatial_shape}, got {spatial_features.shape}"


def test_different_gnn_types():
    """Test extractor with different GNN types to ensure compatibility."""
    batch_size = 1
    height, width = 8, 8

    observation_space = spaces.Dict(
        {
            "grid": spaces.Box(0, 1, (2, height, width)),
            "node_features": spaces.Box(-1, 1, (5, 4)),
            "edge_index": spaces.Box(0, 4, (2, 5), dtype=int),
            "current_node": spaces.Discrete(5),
        }
    )

    cnn_config = {"hidden_channels": (8,), "kernel_sizes": (3,)}
    fusion_config = {"fusion_channels": (8,), "fusion_kernel_sizes": (3,)}

    for gnn_type in ["GCN", "GraphSAGE", "GAT"]:
        gnn_config = {"hidden_dim": 8, "num_layers": 1, "gnn_type": gnn_type}

        extractor = ChipPlacementFeaturesExtractor(
            observation_space=observation_space,
            gnn_config=gnn_config,
            cnn_config=cnn_config,
            fusion_config=fusion_config,
        )

        observations = {
            "grid": torch.randn(batch_size, 2, height, width),
            "node_features": torch.randn(batch_size, 5, 4),
            "edge_index": torch.randint(0, 5, (batch_size, 2, 5)),
            "current_node": torch.randint(0, 5, (batch_size,)),
        }

        output = extractor(observations)
        assert output is not None
