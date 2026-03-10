"""Tests for DiffusionUnetPolicy."""

import torch

from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy


def _make_tiny_policy(**kwargs) -> DiffusionUnetPolicy:
    """Create a tiny policy for fast testing."""
    defaults = dict(
        action_dim=4,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        lowdim_obs_dim=9,
        n_diffusion_steps_train=10,
        n_diffusion_steps_infer=4,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=2,
        diffusion_step_embed_dim=8,
        vision_feature_dim=16,
        crop_shape=(76, 76),
        pretrained_vision=False,
        camera_names=("agentview",),
    )
    defaults.update(kwargs)
    return DiffusionUnetPolicy(**defaults)


def _make_batch(B: int = 2, with_images: bool = True) -> dict[str, torch.Tensor]:
    """Create a fake batch for testing."""
    batch: dict[str, torch.Tensor] = {
        "lowdim_obs": torch.randn(B, 2, 9),
        "actions": torch.randn(B, 16, 4),
    }
    if with_images:
        batch["agentview_image"] = torch.randn(B, 2, 3, 84, 84)
    return batch


class TestDiffusionUnetPolicy:
    def test_compute_loss_is_scalar(self) -> None:
        policy = _make_tiny_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_compute_loss_gradient_flow(self) -> None:
        policy = _make_tiny_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()

        # Check that gradients exist for U-Net parameters
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.unet.parameters()
        )
        assert has_grads

    def test_predict_action_shape_batched(self) -> None:
        policy = _make_tiny_policy()
        obs = {
            "lowdim_obs": torch.randn(2, 2, 9),
            "agentview_image": torch.randn(2, 2, 3, 84, 84),
        }
        actions = policy.predict_action(obs)
        assert actions.shape == (2, 8, 4)  # (B, action_horizon, action_dim)

    def test_predict_action_shape_unbatched(self) -> None:
        policy = _make_tiny_policy()
        obs = {
            "lowdim_obs": torch.randn(2, 9),
            "agentview_image": torch.randn(2, 3, 84, 84),
        }
        actions = policy.predict_action(obs)
        assert actions.shape == (8, 4)  # (action_horizon, action_dim)

    def test_predict_action_no_images(self) -> None:
        policy = _make_tiny_policy(camera_names=())
        # Recalculate global_cond_dim for no cameras
        policy = DiffusionUnetPolicy(
            action_dim=4,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            lowdim_obs_dim=9,
            n_diffusion_steps_train=10,
            n_diffusion_steps_infer=4,
            down_dims=(8, 16),
            kernel_size=3,
            n_groups=2,
            diffusion_step_embed_dim=8,
            vision_feature_dim=16,
            pretrained_vision=False,
            camera_names=(),
        )
        obs = {"lowdim_obs": torch.randn(1, 2, 9)}
        actions = policy.predict_action(obs)
        assert actions.shape == (1, 8, 4)

    def test_set_normalizer(self) -> None:
        import numpy as np

        policy = _make_tiny_policy()
        norm = LinearNormalizer()
        norm.fit({
            "actions": np.random.randn(100, 4).astype(np.float32),
            "lowdim_obs": np.random.randn(100, 9).astype(np.float32),
        })
        policy.set_normalizer(norm)
        assert policy.normalizer is not None

    def test_predict_with_normalizer(self) -> None:
        import numpy as np

        policy = _make_tiny_policy()
        norm = LinearNormalizer()
        norm.fit({
            "actions": np.random.randn(100, 4).astype(np.float32),
            "lowdim_obs": np.random.randn(100, 9).astype(np.float32),
        })
        policy.set_normalizer(norm)

        obs = {
            "lowdim_obs": torch.randn(1, 2, 9),
            "agentview_image": torch.randn(1, 2, 3, 84, 84),
        }
        actions = policy.predict_action(obs)
        assert actions.shape == (1, 8, 4)
        assert torch.isfinite(actions).all()

    def test_training_reduces_loss(self) -> None:
        """A few training steps should reduce loss."""
        policy = _make_tiny_policy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        batch = _make_batch(B=4)
        losses = []

        for _ in range(10):
            loss = policy.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (use first few vs last few average)
        assert sum(losses[:3]) / 3 > sum(losses[-3:]) / 3
