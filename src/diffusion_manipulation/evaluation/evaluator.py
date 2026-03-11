"""Rollout evaluator for diffusion policy."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from diffusion_manipulation.env.video_recorder import VideoRecorder


@dataclass
class EvalResult:
    """Results from evaluation runs."""

    success_rate: float
    num_episodes: int
    num_successes: int
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)
    seed: int = 0


@dataclass
class MultiSeedResult:
    """Aggregated results across multiple seeds."""

    mean_success_rate: float
    std_success_rate: float
    per_seed_results: list[EvalResult] = field(default_factory=list)


def evaluate_policy(
    policy: torch.nn.Module,
    env,
    num_episodes: int = 50,
    obs_horizon: int = 2,
    action_horizon: int = 8,
    max_episode_steps: int = 400,
    camera_names: tuple[str, ...] = ("agentview",),
    device: torch.device | None = None,
    video_recorder: VideoRecorder | None = None,
    record_episodes: int = 0,
) -> EvalResult:
    """Evaluate a policy with receding-horizon rollouts.

    Uses observation deque (maxlen=obs_horizon) and executes
    action_horizon actions per policy call.

    Args:
        policy: Policy with predict_action() method.
        env: Environment with reset/step/get_obs/check_success.
        num_episodes: Number of episodes to evaluate.
        obs_horizon: Number of observation steps to keep.
        action_horizon: Number of actions to execute per prediction.
        max_episode_steps: Maximum steps per episode.
        camera_names: Camera names for image observations.
        device: Device to run policy on.
        video_recorder: Optional recorder for saving rollout videos.
        record_episodes: Number of episodes to record (0 = none).

    Returns:
        EvalResult with success rate and episode stats.
    """
    device = device or torch.device("cpu")
    policy.eval()

    successes = 0
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_successes: list[bool] = []

    for ep_idx in range(num_episodes):
        recording = video_recorder is not None and ep_idx < record_episodes
        if recording:
            video_recorder.start()

        obs = env.reset()
        obs_deque: deque[dict[str, npt.NDArray]] = deque(maxlen=obs_horizon)
        obs_deque.append(obs)

        # Pad initial observations by repeating first obs
        while len(obs_deque) < obs_horizon:
            obs_deque.appendleft(obs)

        total_reward = 0.0
        step_count = 0
        done = False
        episode_success = False

        while step_count < max_episode_steps and not done:
            # Build observation tensors from deque
            obs_dict = _build_obs_tensor(obs_deque, camera_names, device)

            # Predict action sequence
            with torch.no_grad():
                action_seq = policy.predict_action(obs_dict)

            if action_seq.dim() == 3:
                action_seq = action_seq.squeeze(0)

            # Execute action_horizon actions
            actions_np = action_seq.cpu().numpy()
            for action_idx in range(min(action_horizon, len(actions_np))):
                if step_count >= max_episode_steps or done:
                    break

                action = actions_np[action_idx]
                obs, reward, done, info = env.step(action)
                obs_deque.append(obs)
                total_reward += reward
                step_count += 1

                # Check success at every step (standard robomimic protocol)
                if env.check_success():
                    episode_success = True

                if recording:
                    for cam_name in camera_names:
                        img_key = f"{cam_name}_image"
                        if img_key in obs:
                            video_recorder.add_frame(obs[img_key])

        if episode_success:
            successes += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_successes.append(episode_success)

        if recording:
            video_recorder.stop()

    return EvalResult(
        success_rate=successes / num_episodes,
        num_episodes=num_episodes,
        num_successes=successes,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_successes=episode_successes,
    )


def evaluate_multi_seed(
    policy: torch.nn.Module,
    env_factory,
    seeds: tuple[int, ...] = (42, 123, 456),
    num_episodes: int = 50,
    **kwargs,
) -> MultiSeedResult:
    """Evaluate policy across multiple seeds.

    Args:
        policy: Policy to evaluate.
        env_factory: Callable(seed) that creates a new env.
        seeds: Random seeds for evaluation.
        num_episodes: Episodes per seed.
        **kwargs: Additional args passed to evaluate_policy.

    Returns:
        MultiSeedResult with mean±std success rate.
    """
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = env_factory(seed)
        result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=num_episodes,
            **kwargs,
        )
        result.seed = seed
        results.append(result)
        env.close()

    rates = [r.success_rate for r in results]
    return MultiSeedResult(
        mean_success_rate=float(np.mean(rates)),
        std_success_rate=float(np.std(rates)),
        per_seed_results=results,
    )


def _build_obs_tensor(
    obs_deque: deque[dict[str, npt.NDArray]],
    camera_names: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build batched observation tensors from observation deque."""
    obs_list = list(obs_deque)

    # Stack low-dim obs: (To, D) -> (1, To, D)
    lowdim_stack = np.stack([o["lowdim"] for o in obs_list], axis=0)
    result: dict[str, torch.Tensor] = {
        "lowdim_obs": torch.from_numpy(lowdim_stack).float().unsqueeze(0).to(device),
    }

    # Stack images: (To, H, W, C) -> (1, To, C, H, W)
    # Flip vertically: training data (mujoco-py/OpenCV convention) has opposite
    # vertical orientation from live robosuite 1.4+ (DeepMind mujoco/OpenGL)
    for cam_name in camera_names:
        img_key = f"{cam_name}_image"
        if img_key in obs_list[0]:
            imgs = np.stack([o[img_key] for o in obs_list], axis=0)
            imgs = imgs[:, ::-1, :, :]  # Flip H dimension
            imgs_tensor = torch.from_numpy(imgs.copy()).float() / 255.0
            imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)  # (To, C, H, W)
            result[img_key] = imgs_tensor.unsqueeze(0).to(device)

    return result
