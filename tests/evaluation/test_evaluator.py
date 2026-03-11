"""Tests for policy evaluator using mock environment."""

import numpy as np
import torch

from diffusion_manipulation.evaluation.evaluator import (
    EvalResult,
    MultiSeedResult,
    evaluate_multi_seed,
    evaluate_policy,
    _build_obs_tensor,
)
from collections import deque


class MockEnv:
    """Mock environment for testing evaluation logic."""

    def __init__(self, episode_length: int = 10, success_rate: float = 0.5, seed: int = 0) -> None:
        self.episode_length = episode_length
        self.success_rate = success_rate
        self._step = 0
        self._rng = np.random.default_rng(seed)
        self._success = False

    def reset(self) -> dict[str, np.ndarray]:
        self._step = 0
        self._success = self._rng.random() < self.success_rate
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._step += 1
        done = self._step >= self.episode_length
        reward = 1.0 if self._success and done else 0.0
        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "lowdim": np.zeros(9, dtype=np.float32),
            "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
        }

    def check_success(self) -> bool:
        return self._success

    def close(self) -> None:
        pass


class MockPolicy(torch.nn.Module):
    """Mock policy that returns random actions."""

    def __init__(self, action_dim: int = 7, action_horizon: int = 8) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        # Need at least one parameter for device detection
        self.dummy = torch.nn.Linear(1, 1)

    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        B = obs["lowdim_obs"].shape[0]
        return torch.randn(B, self.action_horizon, self.action_dim)


class TestEvaluatePolicy:
    def test_basic_evaluation(self) -> None:
        policy = MockPolicy()
        env = MockEnv(episode_length=5, success_rate=1.0)

        result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=5,
            obs_horizon=2,
            action_horizon=8,
            max_episode_steps=100,
            camera_names=("agentview",),
        )

        assert isinstance(result, EvalResult)
        assert result.num_episodes == 5
        assert len(result.episode_rewards) == 5
        assert len(result.episode_lengths) == 5
        assert len(result.episode_successes) == 5
        assert all(isinstance(s, bool) for s in result.episode_successes)

    def test_success_rate_calculation(self) -> None:
        policy = MockPolicy()
        env = MockEnv(episode_length=5, success_rate=1.0, seed=42)

        result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=10,
            obs_horizon=2,
            action_horizon=8,
            max_episode_steps=100,
            camera_names=("agentview",),
        )

        assert result.success_rate == result.num_successes / result.num_episodes

    def test_max_episode_steps(self) -> None:
        policy = MockPolicy()
        env = MockEnv(episode_length=1000)  # Very long episodes

        result = evaluate_policy(
            policy=policy,
            env=env,
            num_episodes=2,
            obs_horizon=2,
            action_horizon=8,
            max_episode_steps=20,
            camera_names=("agentview",),
        )

        assert all(length <= 20 for length in result.episode_lengths)


class TestEvaluateMultiSeed:
    def test_multi_seed_evaluation(self) -> None:
        policy = MockPolicy()

        def env_factory(seed: int) -> MockEnv:
            return MockEnv(episode_length=5, success_rate=0.5, seed=seed)

        result = evaluate_multi_seed(
            policy=policy,
            env_factory=env_factory,
            seeds=(42, 123),
            num_episodes=5,
            obs_horizon=2,
            action_horizon=8,
            max_episode_steps=100,
            camera_names=("agentview",),
        )

        assert isinstance(result, MultiSeedResult)
        assert len(result.per_seed_results) == 2
        assert 0.0 <= result.mean_success_rate <= 1.0
        assert result.std_success_rate >= 0.0


class TestBuildObsTensor:
    def test_builds_correct_shapes(self) -> None:
        obs = {
            "lowdim": np.zeros(9, dtype=np.float32),
            "agentview_image": np.zeros((84, 84, 3), dtype=np.uint8),
        }
        obs_deque = deque([obs, obs], maxlen=2)

        result = _build_obs_tensor(obs_deque, ("agentview",), torch.device("cpu"))

        assert result["lowdim_obs"].shape == (1, 2, 9)
        assert result["agentview_image"].shape == (1, 2, 3, 84, 84)
        assert result["agentview_image"].dtype == torch.float32
        assert result["agentview_image"].max() <= 1.0
