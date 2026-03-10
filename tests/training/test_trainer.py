"""Tests for the training loop."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from diffusion_manipulation.config import TrainConfig
from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.policy.diffusion_policy import DiffusionUnetPolicy
from diffusion_manipulation.training.trainer import Trainer


def _make_tiny_setup(tmp_path: Path, num_samples: int = 16):
    """Create minimal policy, dataloader, normalizer for training tests."""
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

    # Create synthetic dataset (no images for speed)
    lowdim = torch.randn(num_samples, 2, 9)
    actions = torch.randn(num_samples, 16, 4)

    dataset = TensorDataset(lowdim, actions)

    class WrappedLoader:
        """Wraps DataLoader to yield dicts instead of tuples."""

        def __init__(self, loader):
            self._loader = loader

        def __iter__(self):
            for lowdim_batch, actions_batch in self._loader:
                yield {"lowdim_obs": lowdim_batch, "actions": actions_batch}

        def __len__(self):
            return len(self._loader)

    loader = WrappedLoader(DataLoader(dataset, batch_size=8, shuffle=True))

    normalizer = LinearNormalizer()
    normalizer.fit({
        "actions": actions.numpy().reshape(-1, 4),
        "lowdim_obs": lowdim.numpy().reshape(-1, 9),
    })

    config = TrainConfig(
        num_epochs=3,
        batch_size=8,
        lr=1e-3,
        checkpoint_dir=tmp_path / "checkpoints",
        log_interval=1,
        save_interval=2,
        use_wandb=False,
        num_workers=0,
    )

    return policy, loader, normalizer, config


class TestTrainer:
    def test_training_completes(self, tmp_path: Path) -> None:
        policy, loader, normalizer, config = _make_tiny_setup(tmp_path)
        trainer = Trainer(policy, loader, normalizer, config, device=torch.device("cpu"))

        history = trainer.train()

        assert len(history["loss"]) == 3
        assert all(isinstance(l, float) for l in history["loss"])

    def test_loss_decreases(self, tmp_path: Path) -> None:
        policy, loader, normalizer, config = _make_tiny_setup(tmp_path, num_samples=32)
        config = TrainConfig(
            num_epochs=10,
            batch_size=8,
            lr=1e-3,
            checkpoint_dir=tmp_path / "checkpoints",
            log_interval=100,
            save_interval=100,
            use_wandb=False,
            num_workers=0,
        )

        trainer = Trainer(policy, loader, normalizer, config, device=torch.device("cpu"))
        history = trainer.train()

        # Average of first 3 epochs should be higher than last 3
        early = sum(history["loss"][:3]) / 3
        late = sum(history["loss"][-3:]) / 3
        assert early > late

    def test_checkpoint_save_load(self, tmp_path: Path) -> None:
        policy, loader, normalizer, config = _make_tiny_setup(tmp_path)
        trainer = Trainer(policy, loader, normalizer, config, device=torch.device("cpu"))

        # Train a bit
        trainer.train()

        # Save checkpoint
        ckpt_path = trainer.save_checkpoint("test_checkpoint.pt")
        assert ckpt_path.exists()

        # Create new trainer and load
        policy2, loader2, normalizer2, config2 = _make_tiny_setup(tmp_path / "new")
        trainer2 = Trainer(policy2, loader2, normalizer2, config2, device=torch.device("cpu"))
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.epoch == trainer.epoch
        assert trainer2.global_step == trainer.global_step

    def test_history_saved(self, tmp_path: Path) -> None:
        import json

        policy, loader, normalizer, config = _make_tiny_setup(tmp_path)
        trainer = Trainer(policy, loader, normalizer, config, device=torch.device("cpu"))
        trainer.train()

        history_path = config.checkpoint_dir / "training_history.json"
        assert history_path.exists()

        with open(history_path) as f:
            history = json.load(f)
        assert "loss" in history
        assert "lr" in history
