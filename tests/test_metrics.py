import numpy as np
import torch

from env.metrics import calculate_incremental_wire_length, calculate_total_wirelength


def test_total_wirelength():
    positions = np.array(
        [
            [0, 0],  # Node 0 at (0, 0)
            [0, 5],  # Node 1 at (0, 5)
            [0, 10],  # Node 2 at (0, 10)
        ]
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],  # src nodes
            [1, 0, 2, 1],  # dst nodes (bidirectional)
        ],
        dtype=torch.long,
    )

    total_wl = calculate_total_wirelength(positions, edge_index)
    print(f"Total wirelength: {total_wl}")
    print(f"Expected: {5 + 5 + 5 + 5} = 20 (each edge counted twice)")

    # Test case 1: Empty graph
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[], []], dtype=torch.long)
    assert calculate_total_wirelength(positions, edge_index) == 0.0

    # Test case 2: Simple graph
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    assert calculate_total_wirelength(positions, edge_index) == 2.0

    # Test case 3: More complex graph
    positions = np.array([[0, 0], [1, 1], [2, 2]])
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    assert calculate_total_wirelength(positions, edge_index) == 4.0

    # Test case 4: Different norm
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    assert calculate_total_wirelength(positions, edge_index, norm="euclidean") == 1.4142135623730951

    print("All tests passed!")


def test_incremental_wirelength():
    # Test incremental
    placed = {0: (0, 0), 1: (0, 5)}

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],  # src nodes
            [1, 0, 2, 1],  # dst nodes (bidirectional)
        ],
        dtype=torch.long,
    )

    incremental_wl = calculate_incremental_wire_length(
        node_idx=2, position=(0, 10), placed_positions=placed, edge_index=edge_index
    )
    print(f"\nIncremental wirelength for node 2: {incremental_wl}")
    print("Expected: 5 (only edge to node 1)")

    print("All tests passed!")
