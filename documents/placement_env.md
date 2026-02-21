# Placement Environment

`ChipPlacementEnv` is a custom Gymnasium-compatible environment for representing the chip placement problem as a Markov Decision Process (MDP).

## Overview

The environment represents chip placement as a sequence of decisions. In each step, the agent chooses a position on a 2D grid for the next component in the netlist.

## Environment State Machine

```mermaid
state_Diagram-v2
    [*] --> Reset
    Reset --> GetObservation: Initial State
    GetObservation --> ChooseAction: Agent Decision
    ChooseAction --> Step: Apply Action
    Step --> CheckConstraints
    CheckConstraints --> GetObservation: Valid / Not Done
    CheckConstraints --> [*]: Done (Terminated/Truncated)
    CheckConstraints --> Penalty: Illegal Placement
    Penalty --> [*]
```

## Reward Function

The reward is designed to guide the agent toward minimizing total wirelength:
- **Success**: Negative incremental wirelength (to minimize).
- **Failure**: Large negative penalty for illegal placements (overlap/out of bounds).

## Testing

The script tests/test_placement_env.py verifies:
1. Environment reset and initialization.
2. Step logic and reward calculation.
3. Action masking (suggesting only valid positions).
4. Full episode execution walkthrough.
