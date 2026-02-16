import networkx as nx
import numpy as np
import torch


class NetlistGenerator:
    """Generates a random netlist for the chip placement environment."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_erdos_renyi_graph(
        self, num_nodes: int, edge_probability: float, node_feature_dim: int = 8
    ) -> tuple[torch.Tensor, torch.Tensor, nx.Graph]:
        """
        Generate Erdos-Renyi random graph.

        Args:
            num_nodes: Number of components
            edge_probability: Probability of edge between any two nodes
            node_feature_dim: Dimension of random node features

        Returns:
            node_features: (num_nodes, node_feature_dim)
            edge_index: (2, num_edges)
            nx_graph: NetworkX graph for visualization
        """
        G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability, seed=self.rng)

        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node_a = self.rng.choice(list(components[i]))
                node_b = self.rng.choice(list(components[i + 1]))
                G.add_edge(node_a, node_b)

        node_features = torch.randn(num_nodes, node_feature_dim)

        edge_list = list(G.edges())
        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edges = []
            for u, v in edge_list:
                edges.append([u, v])
                edges.append([v, u])
            edge_index = torch.tensor(edges, dtype=torch.long).t()

        return node_features, edge_index, G
