"""Shared test fixtures."""

from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def tmp_hdf5(tmp_path: Path) -> Path:
    """Create a synthetic robomimic-style HDF5 dataset with 3 episodes."""
    hdf5_path = tmp_path / "test_dataset.hdf5"

    episode_lengths = [10, 15, 8]
    action_dim = 7
    lowdim_dims = {"robot0_eef_pos": 3, "robot0_eef_quat": 4, "robot0_gripper_qpos": 2}
    img_shape = (84, 84, 3)

    rng = np.random.default_rng(42)

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")

        for ep_idx, ep_len in enumerate(episode_lengths):
            demo = data_grp.create_group(f"demo_{ep_idx}")
            demo.create_dataset("actions", data=rng.standard_normal((ep_len, action_dim)).astype(np.float32))
            demo.attrs["num_samples"] = ep_len

            obs = demo.create_group("obs")
            for key, dim in lowdim_dims.items():
                obs.create_dataset(key, data=rng.standard_normal((ep_len, dim)).astype(np.float32))

            obs.create_dataset(
                "agentview_image",
                data=rng.integers(0, 255, size=(ep_len, *img_shape), dtype=np.uint8),
            )

    return hdf5_path


@pytest.fixture
def tmp_hdf5_no_images(tmp_path: Path) -> Path:
    """Create a synthetic HDF5 dataset without image observations."""
    hdf5_path = tmp_path / "test_lowdim.hdf5"

    episode_lengths = [5, 7]
    action_dim = 7
    rng = np.random.default_rng(42)

    with h5py.File(hdf5_path, "w") as f:
        data_grp = f.create_group("data")

        for ep_idx, ep_len in enumerate(episode_lengths):
            demo = data_grp.create_group(f"demo_{ep_idx}")
            demo.create_dataset("actions", data=rng.standard_normal((ep_len, action_dim)).astype(np.float32))

            obs = demo.create_group("obs")
            obs.create_dataset("robot0_eef_pos", data=rng.standard_normal((ep_len, 3)).astype(np.float32))
            obs.create_dataset("robot0_eef_quat", data=rng.standard_normal((ep_len, 4)).astype(np.float32))
            obs.create_dataset("robot0_gripper_qpos", data=rng.standard_normal((ep_len, 2)).astype(np.float32))

    return hdf5_path
