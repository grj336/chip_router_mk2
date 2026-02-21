# Metrics

The `metrics.py` module provides functions to evaluate the quality of a chip placement, primarily focusing on wirelength.

## Overview

Wirelength is the primary metric for optimization in chip placement. Shorter wirelengths generally lead to lower power consumption, higher performance, and less congestion.

## Wirelength Calculation Logic

```mermaid
graph LR
    A[Source Position] -- Difference --> B[Delta X, Delta Y]
    B -- Manhattan --> C[Sum of Absolutes]
    B -- Euclidean --> D[L2 Norm]
    C --> E[Edge Length]
    D --> E
    E --> F[Sum All Edges]
    F --> G[Total Wirelength]
```

## Key Functions

- `calculate_total_wirelength(positions, edge_index, norm)`: Computes the total wirelength of a fully placed netlist.
- `calculate_incremental_wire_length(node_idx, position, placed_positions, edge_index, norm)`: Computes the wirelength added by placing a single node into an existing partial placement.

## Testing

The script tests/test_metrics.py verifies:
1. Correctness of Manhattan and Euclidean distance calculations.
2. Handling of empty graphs.
3. Accurate incremental updates as nodes are placed.
