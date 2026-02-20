# Metrics to eval chip placement

import numpy as np
import torch


def calculate_total_wirelength(
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


def calculate_incremental_wire_length(
    node_idx: int,
    position: tuple[int, int],
    placed_positions: dict[int, tuple[int, int]],
    edge_index: torch.tensor,
    norm: str = "manhattan",
) -> float:
    """Calculate incremental wirelength of placing a node.

    Args:
        node_idx: Index of the node to place.
        positions: Position (x, y) if current node.
        placed_positions: Dictionary of placed node positions.
        edge_index: Array of shape (2, num_edges) array of edges between nodes.
        norm: Normalization method, either "manhattan" or "euclidean".

    Returns:
        Incremental wirelength: Sum of edge lengths to placed neighbours
    """
    if edge_index.shape[1] == 0:
        return 0.0

    edge_index_np = edge_index.cpu().numpy()

    # Find edges connected to the current node
    outgoing_mask = edge_index_np[0] == node_idx
    incoming_mask = edge_index_np[1] == node_idx

    total_wirelength = 0.0

    # Check outgoing edges
    for i in np.where(outgoing_mask)[0]:
        neighbor_idx = int(edge_index_np[1, i])
        if neighbor_idx in placed_positions:
            neighbor_pos = placed_positions[neighbor_idx]
            if norm == "manhattan":
                dist = abs(position[0] - neighbor_pos[0]) + abs(position[1] - neighbor_pos[1])
            elif norm == "euclidean":
                dist = np.linalg.norm(np.array(position) - np.array(neighbor_pos), ord=2)
            else:
                dist = np.sqrt(
                    (position[0] - neighbor_pos[0]) ** 2 + (position[1] - neighbor_pos[1]) ** 2
                )
            total_wirelength += dist

    # Check incoming edges
    for i in np.where(incoming_mask)[0]:
        neighbor_idx = int(edge_index_np[0, i])
        if neighbor_idx in placed_positions:
            neighbor_pos = placed_positions[neighbor_idx]
            if norm == "manhattan":
                dist = abs(position[0] - neighbor_pos[0]) + abs(position[1] - neighbor_pos[1])
            elif norm == "euclidean":
                dist = np.linalg.norm(np.array(position) - np.array(neighbor_pos), ord=2)
            else:
                dist = np.sqrt(
                    (position[0] - neighbor_pos[0]) ** 2 + (position[1] - neighbor_pos[1]) ** 2
                )
            total_wirelength += dist

    return total_wirelength
