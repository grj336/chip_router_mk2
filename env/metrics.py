# Metrics to eval chip placement


import numpy as np
import torch


def total_wirelength(
    positions: np.ndarray, edge_index: torch.tensor, norm: str = "manhattan"
) -> float:
    """Calculate total wirelength of placements.

    Args:
        positions: Array of shape (num_nodes, 2) array of (x, y) coordinates of each node.
        edge_index: Array of shape (2, num_edges) array of edges between nodes.
        norm: Normalization method, either "manhattan" or "euclidean".

    Returns:
        Total wirelength. Sum of wire lengths
    """

    if edge_index.shape[1] == 0:
        return 0.0

    edge_index_np = edge_index.cpu().numpy()

    src_positions = positions[edge_index_np[0]]  # (num_edges, 2)
    dst_positions = positions[edge_index_np[1]]  # (num_edges, 2)

    if norm == "manhattan":
        distances = np.abs(src_positions - dst_positions).sum(axis=1)
    elif norm == "euclidean":
        distances = np.linalg.norm(src_positions - dst_positions, ord=2, axis=1)
    else:
        raise ValueError(f"Unknown norm: {norm}")

    return float(distances.sum())
