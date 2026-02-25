import sys
from pathlib import Path

import numpy as np
import torch

# Add parent dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.chip_env import ChipEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import MaskablePPO

from config import ExperimentConfig
from env import ChipPlacementEnv
from models import ChipPlacementFusionPolicy


def make_env(config: ExperimentConfig, rank: int = 0):
    """Make environment for training"""

    def _init() -> ChipEnv:
        env = ChipPlacementEnv(
            grid_height=config.env.grid_height,
            grid_width=config.env.grid_width,
            num_components_min=config.env.num_components_min,
            num_components_max=config.env.num_components_max,
            edge_probability=config.env.edge_probability,
            node_feature_dim=config.env.node_feature_dim,
            wirelength_weight=config.env.wirelength_weight,
            illegal_placement_penalty=config.env.illegal_placement_penalty,
            seed=config.seed + rank,
        )
        env = Monitor(make_env)
        return env

    return _init


def create_vec_env(config: ExperimentConfig, n_envs: int = 4):
    """Create vectorized environment"""
    if n_envs == 1:
        return DummyVecEnv([make_env(config, 0)])
    else:
        # Use subprocess for parallelism
        return SubprocVecEnv([make_env(config, i) for i in range(n_envs)])


def train(
    config: ExperimentConfig,
    time_timesteps: int = 50000,
    n_envs: int = 4,
    save_freq: int = 10000,
    eval_freq: int = 5000,
) -> MaskablePPO:
    """
    Train the chip placement agent with Maskable PPO

    Args:
        config: Experiment configuration
        time_timesteps: Number of timesteps to train for
        n_envs: Number of parallel environments
        save_freq: Frequency to save the model
        eval_freq: Frequency to evaluate the model

    Returns:
        Trained model
    """

    print("=" * 80)
    print("GNN-RL Chip Placement - Starting training")
    print("=" * 80)
    print(f"\nDevice: {config.device}")
    print(f"Seed: {config.seed}")
    print(f"Total timesteps: {time_timesteps}")
    print(f"Number of parallel environments: {n_envs}")
    print(f"Save frequency: {save_freq}")
    print(f"Evaluation frequency: {eval_freq}")

    # set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create local dirs
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    tensorboard_dir = log_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)

    # Create vectorized environment
    print("Creating vectorized training environment...")
    train_env = create_vec_env(config, n_envs)

    # Create evaluation environment
    print("Creating vectorized evaluation environment...")
    eval_env = create_vec_env(config, n_envs=1)

    # Prepare policy kwargs
    policy_kwargs = {
        "gnn_config": {
            "hidden_dim": config.gnn.hidden_dim,
            "num_layers": config.gnn.num_layers,
            "gnn_type": "GraphSAGE",
            "aggr": "mean",
            "dropout": 0.0,
            "batch_norm": True,
        },
        "cnn_config": {
            "hidden_dim": config.cnn.hidden_dim,
            "kernel_size": config.cnn.kernel_size,
            "dropout": 0.0,
            "batch_norm": True,
        },
        "fusion_config": {
            "fusion_channels": (64, 32),
            "fusion_kernel_sizes": (3, 1),
            "output_channels": 1,
        },
        "net_arch": [256, 256],
    }

    # Create model
    print("\nInitializing MaskablePPO model...")
    print(f"  GNN: GraphSAGE with {config.gnn.num_layers} layers, dim {config.gnn.hidden_dim}")
    print(f"  CNN: {len(config.cnn.hidden_channels)} layers, channels {config.cnn.hidden_channels}")
    print(f"  Grid: {config.env.grid_height}x{config.env.grid_width}")
    print(f"  Components: {config.env.num_components_min}-{config.env.num_components_max}\n")

    # Create model
    # TODO: Add config.rl.policy_kwargs to the model
    model = MaskablePPO(
        policy=ChipPlacementFusionPolicy,
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=config.device,
        tensorboard_log=str(tensorboard_dir),
        seed=config.seed,
    )

    # Set callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix="model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=time_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save model
    final_model_path = log_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")

    print("Training completed!")

    # Clean up
    train_env.close()
    eval_env.close()

    return model
