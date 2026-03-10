"""Replay buffer for loading HDF5 episodes into flat indexed storage."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import numpy.typing as npt


@dataclass
class ReplayBuffer:
    """Flat indexed storage with episode boundary tracking.

    Concatenates all episodes into flat arrays, tracking where each
    episode ends for boundary-aware sequence sampling.
    """

    actions: npt.NDArray[np.float32]
    lowdim_obs: npt.NDArray[np.float32]
    images: dict[str, npt.NDArray[np.uint8]]
    episode_ends: npt.NDArray[np.int64]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    def get_episode_slice(self, episode_idx: int) -> slice:
        """Get the slice for a given episode index."""
        start = 0 if episode_idx == 0 else int(self.episode_ends[episode_idx - 1])
        end = int(self.episode_ends[episode_idx])
        return slice(start, end)

    def sample_sequence(
        self,
        index: int,
        obs_horizon: int,
        pred_horizon: int,
    ) -> dict[str, npt.NDArray]:
        """Sample a sequence window centered at a flat index.

        Returns observation and action sequences with padding at
        episode boundaries (repeats edge values).

        Args:
            index: Flat step index into the concatenated buffer.
            obs_horizon: Number of observation steps to return.
            pred_horizon: Number of action steps to return.

        Returns:
            Dict with 'lowdim_obs', 'actions', and image keys.
        """
        episode_idx = int(np.searchsorted(self.episode_ends, index, side="right"))
        ep_slice = self.get_episode_slice(episode_idx)
        ep_start, ep_end = ep_slice.start, ep_slice.stop
        ep_len = ep_end - ep_start
        local_idx = index - ep_start

        # Observation window: [local_idx - obs_horizon + 1, local_idx + 1)
        obs_start = local_idx - obs_horizon + 1
        obs_end = local_idx + 1

        # Action window: [local_idx, local_idx + pred_horizon)
        act_start = local_idx
        act_end = local_idx + pred_horizon

        result: dict[str, npt.NDArray] = {}

        # Extract with boundary padding
        result["lowdim_obs"] = self._extract_padded(
            self.lowdim_obs[ep_start:ep_end], obs_start, obs_end, ep_len
        )
        result["actions"] = self._extract_padded(
            self.actions[ep_start:ep_end], act_start, act_end, ep_len
        )

        for cam_name, img_data in self.images.items():
            result[cam_name] = self._extract_padded(
                img_data[ep_start:ep_end], obs_start, obs_end, ep_len
            )

        return result

    @staticmethod
    def _extract_padded(
        data: npt.NDArray,
        start: int,
        end: int,
        ep_len: int,
    ) -> npt.NDArray:
        """Extract a slice from data with edge padding for out-of-bounds indices."""
        seq_len = end - start
        result_shape = (seq_len,) + data.shape[1:]
        result = np.empty(result_shape, dtype=data.dtype)

        for i in range(seq_len):
            src_idx = start + i
            src_idx = max(0, min(src_idx, ep_len - 1))
            result[i] = data[src_idx]

        return result


def load_replay_buffer(
    hdf5_path: str | Path,
    camera_names: tuple[str, ...] = ("agentview",),
    max_episodes: int | None = None,
) -> ReplayBuffer:
    """Load a robomimic HDF5 dataset into a ReplayBuffer.

    Args:
        hdf5_path: Path to the HDF5 file.
        camera_names: Camera names to load image observations for.
        max_episodes: Maximum number of episodes to load (None = all).

    Returns:
        A ReplayBuffer with concatenated episode data.
    """
    all_actions: list[npt.NDArray] = []
    all_lowdim: list[npt.NDArray] = []
    all_images: dict[str, list[npt.NDArray]] = {cam: [] for cam in camera_names}
    episode_ends: list[int] = []
    total_steps = 0

    with h5py.File(hdf5_path, "r") as f:
        demos = f["data"]
        demo_keys = sorted(demos.keys(), key=lambda x: int(x.replace("demo_", "")))

        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        for demo_key in demo_keys:
            demo = demos[demo_key]
            actions = np.array(demo["actions"], dtype=np.float32)
            n_steps = len(actions)
            all_actions.append(actions)

            # Build low-dim observation from available keys
            lowdim_parts = []
            obs = demo["obs"]
            for key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
                if key in obs:
                    lowdim_parts.append(np.array(obs[key], dtype=np.float32))

            all_lowdim.append(np.concatenate(lowdim_parts, axis=-1))

            # Load images
            for cam_name in camera_names:
                img_key = f"{cam_name}_image"
                if img_key in obs:
                    all_images[cam_name].append(np.array(obs[img_key], dtype=np.uint8))

            total_steps += n_steps
            episode_ends.append(total_steps)

    return ReplayBuffer(
        actions=np.concatenate(all_actions, axis=0),
        lowdim_obs=np.concatenate(all_lowdim, axis=0),
        images={cam: np.concatenate(imgs, axis=0) for cam, imgs in all_images.items() if imgs},
        episode_ends=np.array(episode_ends, dtype=np.int64),
    )
