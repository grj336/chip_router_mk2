import torch
from gymnasium import spaces

from models.fusion_policy import ChipPlacementFusionPolicy

print("Setting up observations space...")
obs_space = spaces.Dict(
    {
        "grid": spaces.Box(low=0, high=1, shape=(2, 16, 16), dtype=float),
        "node_features": spaces.Box(low=-10, high=10, shape=(15, 8), dtype=float),
        "edge_index": spaces.Box(low=0, high=14, shape=(2, 225), dtype=int),
        "current_node": spaces.Discrete(15),
        "action_mask": spaces.MultiBinary(16 * 16),
    }
)

action_space = spaces.MultiDiscrete([16, 16])

print("Creating Policy...")
policy = ChipPlacementFusionPolicy(
    observation_space=obs_space,
    action_space=action_space,
    lr_schedule=lambda x: 3e-4,
    gnn_config={
        "hidden_dim": 64,
        "num_layers": 3,
        "gnn_type": "GraphSAGE",
        "aggr": "mean",
        "dropout": 0.0,
        "batch_norm": True,
    },
    cnn_config={
        "hidden_channels": (32, 64, 64),
        "kernel_sizes": (3, 3, 3),
        "dropout": 0.0,
        "batch_norm": True,
    },
    fusion_config={
        "fusion_channels": (64, 32),
        "fusion_kernel_sizes": (3, 3),
        "output_channels": 1,
    },
)

print("Policy created successfully.")
print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

# Create dummy obs
print("\nCreating dummy observations...")
batch_size = 2
obs = {
    "grid": torch.randn(batch_size, 2, 16, 16),
    "node_features": torch.randn(batch_size, 15, 8),
    "edge_index": torch.randint(0, 15, (batch_size, 2, 225)),
    "current_node": torch.randint(0, 15, (batch_size,)),
    "action_mask": torch.ones(batch_size, 16 * 16, dtype=torch.float32),
}

# Forward pass
print("\nRunning forward pass...")
with torch.no_grad():
    actions, values, log_probs = policy.forward(obs, deterministic=False)

print("\nOutput shapes:")
print(f"  Actions: {actions.shape} - expected (2, 2)")
print(f"  Values: {values.shape} - expected (2, 1)")
print(f"  Log probs: {log_probs.shape} - expected (2,)")

print("\nSample outputs:")
print(f"  Actions: {actions}")
print(f"  Values: {values.squeeze()}")

print("\n✅ Policy working!")
