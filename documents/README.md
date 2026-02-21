# Chip Router MK2 Documentation

Welcome to the documentation for the Chip Router MK2 project. 

## Structure

This folder contains detailed documentation for each core component:

### Environment (`env/`)
- [Netlist Generator](netlist_generator.md): Synthetic graph generation.
- [Metrics](metrics.md): Wirelength and performance evaluation.
- [Placement Environment](placement_env.md): Gymnasium MDP definition.

### Models (`models/`)
- [GNN Encoder](gnn_encoder.md): Topological feature extraction.
- [CNN Encoder](cnn_encoder.md): Spatial feature extraction.

## Running Tests

All components are accompanied by tests. To run all tests and verify functionality:

```bash
uv run pytest
```
