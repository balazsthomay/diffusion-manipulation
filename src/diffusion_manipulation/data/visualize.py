"""Visualization utilities for dataset demos and action distributions."""

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from diffusion_manipulation.data.replay_buffer import ReplayBuffer


def render_demo_gif(
    replay_buffer: ReplayBuffer,
    episode_idx: int,
    camera_name: str,
    output_path: str | Path,
    fps: int = 20,
) -> Path:
    """Render a demo episode as a GIF from camera images.

    Args:
        replay_buffer: Buffer containing episode data.
        episode_idx: Which episode to render.
        camera_name: Camera key in the images dict.
        output_path: Where to save the GIF.
        fps: Frames per second.

    Returns:
        Path to the saved GIF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ep_slice = replay_buffer.get_episode_slice(episode_idx)
    frames = replay_buffer.images[camera_name][ep_slice]  # (T, H, W, C)

    duration_ms = 1000 / fps
    imageio.mimsave(str(output_path), [frame for frame in frames], duration=duration_ms, loop=0)
    return output_path


def plot_action_distributions(
    replay_buffer: ReplayBuffer,
    output_path: str | Path,
    action_labels: tuple[str, ...] = (
        "x", "y", "z", "rx", "ry", "rz", "gripper"
    ),
) -> Path:
    """Plot histograms of action distributions across all episodes.

    Args:
        replay_buffer: Buffer containing action data.
        output_path: Where to save the plot.
        action_labels: Labels for each action dimension.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    actions = replay_buffer.actions
    n_dims = actions.shape[-1]
    n_labels = min(n_dims, len(action_labels))

    fig, axes = plt.subplots(1, n_dims, figsize=(3 * n_dims, 3))
    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.hist(actions[:, i], bins=50, alpha=0.7, color="steelblue")
        label = action_labels[i] if i < n_labels else f"dim_{i}"
        ax.set_title(label)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    fig.suptitle("Action Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_episode_actions(
    replay_buffer: ReplayBuffer,
    episode_idx: int,
    output_path: str | Path,
    action_labels: tuple[str, ...] = (
        "x", "y", "z", "rx", "ry", "rz", "gripper"
    ),
) -> Path:
    """Plot action trajectories for a single episode.

    Args:
        replay_buffer: Buffer containing episode data.
        episode_idx: Which episode to plot.
        output_path: Where to save the plot.
        action_labels: Labels for each action dimension.

    Returns:
        Path to the saved plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ep_slice = replay_buffer.get_episode_slice(episode_idx)
    actions = replay_buffer.actions[ep_slice]
    n_dims = actions.shape[-1]
    n_labels = min(n_dims, len(action_labels))

    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    timesteps = np.arange(len(actions))
    for i, ax in enumerate(axes):
        label = action_labels[i] if i < n_labels else f"dim_{i}"
        ax.plot(timesteps, actions[:, i], color="steelblue", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Episode {episode_idx} Actions", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
