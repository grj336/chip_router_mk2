"""Test GNN encoder"""

import torch

from env.netlist_generator import NetlistGenerator
from models.gnn_encoder import GNNEncoder


def test_gnn_encoder():
    print("Testing GNN encoder...")
    gnn = GNNEncoder(node_feature_dim=8, hidden_dim=64, num_layers=3, gnn_type="GraphSAGE")

    print("GNN architecture:")
    print(gnn)
    print(f"\nOutput dimension: {gnn.get_output_dim()}")

    # Generate a test graph
    print("\nGenerating test netlist...")
    gen = NetlistGenerator(seed=336)
    node_features, edge_index, G = gen.generate_erdos_renyi_graph(
        num_nodes=10, edge_probability=0.3, node_feature_dim=8
    )

    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        # Get all embeddings
        all_embeddings = gnn(node_features, edge_index)
        print(f"All embeddings shape: {all_embeddings.shape}")

        # Get specific node embedding
        current_node = torch.tensor([0, 5])  # Batch of 2
        selected_embeddings = gnn(node_features, edge_index, current_node)
        print(f"Selected embeddings shape: {selected_embeddings.shape}")

    print("\n✅ GNN encoder working!")

    # Create a simple graph
    node_features = torch.randn(5, 10)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])

    # Create encoder
    encoder = GNNEncoder(node_feature_dim=10, hidden_dim=16, num_layers=2)

    # Test forward pass
    output = encoder(node_features, edge_index)
    assert output.shape == (5, 16)

    # Test with current node
    output_current = encoder(node_features, edge_index, current_node=torch.tensor([0]))
    assert output_current.shape == (1, 16)

    print("GNN encoder tests passed!")
