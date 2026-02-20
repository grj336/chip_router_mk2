"""
GNN encoder for netlist representation
produces node embeddings for each node in the netlist
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        gnn_type: str = "GraphSAGE",
        aggr: str = "mean",
        dropout: float = 0.0,
        batch_norm: bool = True,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        if gnn_type == "GCN":
            self.conv_class = GCNConv
        elif gnn_type == "GraphSAGE":
            self.conv_class = SAGEConv
        elif gnn_type == "GAT":
            self.conv_class = GATConv
        else:
            raise ValueError(f"Invalid GNN type: {gnn_type}")

        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None

        # Input layer
        if gnn_type == "GAT":
            self.convs.append(self.conv_class(node_feature_dim, hidden_dim, heads=1, concat=False))
        else:
            self.convs.append(self.conv_class(node_feature_dim, hidden_dim, aggr=aggr))

        if batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            if gnn_type == "GAT":
                self.convs.append(self.conv_class(hidden_dim, hidden_dim, heads=1, concat=False))
            else:
                self.convs.append(self.conv_class(hidden_dim, hidden_dim, aggr=aggr))

            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_node: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = node_features

        # Apply layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)

            x = self.activation(x)
            x = self.dropout(x)

        # Shape: (N, hidden_dim)
        if current_node is not None:
            # Select current node
            return x[current_node]
        else:
            # Return all
            return x

    def get_output_dim(self) -> int:
        return self.hidden_dim
