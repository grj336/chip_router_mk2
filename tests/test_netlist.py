import matplotlib.pyplot as plt
import networkx as nx

from env.netlist_generator import NetlistGenerator


def test_netlist_generation():
    # Create generator
    gen = NetlistGenerator(seed=42)

    # Generate a small graph
    node_features, edge_index, G = gen.generate_erdos_renyi_graph(
        num_nodes=10, edge_probability=0.3, node_feature_dim=8
    )

    print("Generated graph:")
    print(f"  Nodes: {len(G.nodes())}")
    print(f"  Edges: {len(G.edges())}")
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Connected: {nx.is_connected(G)}")

    # Assertions
    assert len(G.nodes()) == 10
    assert nx.is_connected(G)

    # Visualize
    import matplotlib

    matplotlib.use("Agg")
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
    )
    plt.title("Sample Circuit Netlist (10 nodes)")
    plt.savefig("test_netlist.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved to test_netlist.png")
