"""Tests for DiffusionDataset."""

from pathlib import Path

import numpy as np
import torch

from diffusion_manipulation.data.dataset import DiffusionDataset
from diffusion_manipulation.data.replay_buffer import load_replay_buffer


class TestDiffusionDataset:
    def test_length(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))
        assert len(ds) == 33  # 10 + 15 + 8

    def test_getitem_shapes(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))

        sample = ds[5]
        assert sample["lowdim_obs"].shape == (2, 9)
        assert sample["actions"].shape == (16, 7)
        assert sample["agentview_image"].shape == (2, 3, 84, 84)

    def test_getitem_types(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))

        sample = ds[0]
        assert isinstance(sample["lowdim_obs"], torch.Tensor)
        assert isinstance(sample["actions"], torch.Tensor)
        assert isinstance(sample["agentview_image"], torch.Tensor)
        assert sample["lowdim_obs"].dtype == torch.float32
        assert sample["actions"].dtype == torch.float32
        assert sample["agentview_image"].dtype == torch.float32

    def test_image_range(self, tmp_hdf5: Path) -> None:
        """Images should be in [0, 1] after /255 normalization."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))

        sample = ds[0]
        img = sample["agentview_image"]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_actions_normalized(self, tmp_hdf5: Path) -> None:
        """Actions should be approximately in [-1, 1] range."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))

        # Check multiple samples
        for i in range(0, len(ds), 10):
            sample = ds[i]
            assert sample["actions"].min() >= -1.1  # Allow small tolerance from padding
            assert sample["actions"].max() <= 1.1

    def test_no_images(self, tmp_hdf5_no_images: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5_no_images, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=4, camera_names=("agentview",))

        sample = ds[0]
        assert "lowdim_obs" in sample
        assert "actions" in sample
        assert "agentview_image" not in sample

    def test_custom_normalizer(self, tmp_hdf5: Path) -> None:
        from diffusion_manipulation.data.normalizer import LinearNormalizer

        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        norm = LinearNormalizer()
        norm.fit({"actions": buf.actions, "lowdim_obs": buf.lowdim_obs})

        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=8, normalizer=norm)
        sample = ds[0]
        assert sample["actions"].shape == (8, 7)

    def test_dataloader_compatible(self, tmp_hdf5: Path) -> None:
        """Dataset should work with PyTorch DataLoader."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        ds = DiffusionDataset(buf, obs_horizon=2, pred_horizon=16, camera_names=("agentview",))

        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert batch["lowdim_obs"].shape == (4, 2, 9)
        assert batch["actions"].shape == (4, 16, 7)
        assert batch["agentview_image"].shape == (4, 2, 3, 84, 84)
