"""Tests for replay buffer module."""

from pathlib import Path

import numpy as np
import pytest

from diffusion_manipulation.data.replay_buffer import ReplayBuffer, load_replay_buffer


class TestReplayBuffer:
    def test_load_from_hdf5(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        assert buf.num_episodes == 3
        assert buf.num_steps == 10 + 15 + 8  # 33
        assert buf.actions.shape == (33, 7)
        assert buf.lowdim_obs.shape == (33, 9)  # 3 + 4 + 2
        assert "agentview" in buf.images
        assert buf.images["agentview"].shape == (33, 84, 84, 3)

    def test_episode_ends(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))
        np.testing.assert_array_equal(buf.episode_ends, [10, 25, 33])

    def test_episode_slices(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        s0 = buf.get_episode_slice(0)
        assert s0 == slice(0, 10)

        s1 = buf.get_episode_slice(1)
        assert s1 == slice(10, 25)

        s2 = buf.get_episode_slice(2)
        assert s2 == slice(25, 33)

    def test_max_episodes(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",), max_episodes=2)
        assert buf.num_episodes == 2
        assert buf.num_steps == 25

    def test_load_without_images(self, tmp_hdf5_no_images: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5_no_images, camera_names=("agentview",))
        assert buf.num_episodes == 2
        assert len(buf.images) == 0

    def test_sample_sequence_shapes(self, tmp_hdf5: Path) -> None:
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        sample = buf.sample_sequence(index=5, obs_horizon=2, pred_horizon=16)
        assert sample["lowdim_obs"].shape == (2, 9)
        assert sample["actions"].shape == (16, 7)
        assert sample["agentview"].shape == (2, 84, 84, 3)

    def test_sample_sequence_boundary_padding_start(self, tmp_hdf5: Path) -> None:
        """At the start of an episode, obs should be padded by repeating the first frame."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        # Index 0 is the first step of episode 0
        sample = buf.sample_sequence(index=0, obs_horizon=2, pred_horizon=4)
        # obs_horizon=2 means we need indices [-1, 0] -> clamp to [0, 0]
        np.testing.assert_array_equal(sample["lowdim_obs"][0], sample["lowdim_obs"][1])

    def test_sample_sequence_boundary_padding_end(self, tmp_hdf5: Path) -> None:
        """At the end of an episode, actions should be padded by repeating the last frame."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        # Last step of episode 0 is index 9 (ep_len=10)
        sample = buf.sample_sequence(index=9, obs_horizon=2, pred_horizon=16)
        # Action at index 9 is the last; indices 10-24 should be padded with last value
        last_action = buf.actions[9]
        np.testing.assert_array_equal(sample["actions"][-1], last_action)

    def test_sample_at_episode_boundary(self, tmp_hdf5: Path) -> None:
        """Index at the start of episode 1 (index 10) should not cross into episode 0."""
        buf = load_replay_buffer(tmp_hdf5, camera_names=("agentview",))

        sample = buf.sample_sequence(index=10, obs_horizon=2, pred_horizon=4)
        # obs should be episode 1's first step padded
        ep1_first = buf.lowdim_obs[10]
        np.testing.assert_array_equal(sample["lowdim_obs"][0], ep1_first)


class TestReplayBufferDirect:
    """Test ReplayBuffer constructed directly (not from HDF5)."""

    def test_properties(self) -> None:
        buf = ReplayBuffer(
            actions=np.zeros((20, 7), dtype=np.float32),
            lowdim_obs=np.zeros((20, 9), dtype=np.float32),
            images={},
            episode_ends=np.array([10, 20], dtype=np.int64),
        )
        assert buf.num_episodes == 2
        assert buf.num_steps == 20
