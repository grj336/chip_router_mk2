import numpy as np

from env import ChipPlacementEnv

print("Creating environment...")
env = ChipPlacementEnv(
    grid_height=16, grid_width=16, num_components_min=5, num_components_max=5, seed=336
)

print("Resetting environment...")
obs, info = env.reset(seed=336)

print(f"\nObservation space keys: {obs.keys()}")
print(f"Grid shape: {obs['grid'].shape}")
print(f"Node features shape: {obs['node_features'].shape}")
print(f"Current node: {obs['current_node']}")
print(f"Action mask shape: {obs['action_mask'].shape}")
print(f"Valid actions: {obs['action_mask'].sum()}")
print(f"\nInfo: {info}")

print("\n" + "=" * 50 + "\n")
print("Running episodes with random actions...")
print("=" * 50)

done = False
step = 0
total_reward = 0.0

while not done:
    # Get actions
    action_mask = env.action_masks()
    valid_positions = np.where(action_mask)[0]

    action_idx = np.random.choice(valid_positions)
    action = np.array([action_idx // env.grid_width, action_idx % env.grid_width])

    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    step += 1

    print(f"Step {step}, Reward {reward:.2f}, Total Reward {total_reward:.2f}")

print(f"\n{'='*60}")
print("Episode finished!")
print(f"Total steps: {step}")
print(f"Total reward: {total_reward:.2f}")
print(f"Success: {not info['illegal_placement']}")
print(f"{'='*60}")
