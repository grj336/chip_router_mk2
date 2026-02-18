import numpy as np
import torch

from env.metrics import total_wirelength


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

    total_wl = total_wirelength(positions, edge_index)
    print(f"Total wirelength: {total_wl}")
    print(f"Expected: {5 + 5 + 5 + 5} = 20 (each edge counted twice)")

    # Test case 1: Empty graph
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[], []], dtype=torch.long)
    assert total_wirelength(positions, edge_index) == 0.0

    # Test case 2: Simple graph
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    assert total_wirelength(positions, edge_index) == 2.0

    # Test case 3: More complex graph
    positions = np.array([[0, 0], [1, 1], [2, 2]])
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    assert total_wirelength(positions, edge_index) == 4.0

    # Test case 4: Different norm
    positions = np.array([[0, 0], [1, 1]])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    assert total_wirelength(positions, edge_index, norm="euclidean") == 1.4142135623730951

    print("All tests passed!")


if __name__ == "__main__":
    test_total_wirelength()
