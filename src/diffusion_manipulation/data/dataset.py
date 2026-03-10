"""PyTorch Dataset for diffusion policy training."""

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.data.replay_buffer import ReplayBuffer


class DiffusionDataset(Dataset):
    """Dataset that maps flat indices to observation-action horizon windows.

    Each sample contains:
        - obs.images: (To, C, H, W) float32 tensor in [0, 1]
        - obs.lowdim: (To, D) float32 tensor normalized to [-1, 1]
        - action: (Tp, action_dim) float32 tensor normalized to [-1, 1]

    where To = obs_horizon, Tp = pred_horizon.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        camera_names: tuple[str, ...] = ("agentview",),
        normalizer: LinearNormalizer | None = None,
    ) -> None:
        self.replay_buffer = replay_buffer
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.camera_names = camera_names

        # Fit normalizer if not provided
        if normalizer is None:
            normalizer = LinearNormalizer()
            normalizer.fit({
                "actions": replay_buffer.actions,
                "lowdim_obs": replay_buffer.lowdim_obs,
            })
        self.normalizer = normalizer

    def __len__(self) -> int:
        return self.replay_buffer.num_steps

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.replay_buffer.sample_sequence(
            index=index,
            obs_horizon=self.obs_horizon,
            pred_horizon=self.pred_horizon,
        )

        # Normalize low-dim observations
        lowdim = torch.from_numpy(sample["lowdim_obs"]).float()
        lowdim = self.normalizer.normalize("lowdim_obs", lowdim)

        # Normalize actions
        actions = torch.from_numpy(sample["actions"]).float()
        actions = self.normalizer.normalize("actions", actions)

        result: dict[str, torch.Tensor] = {
            "lowdim_obs": lowdim,
            "actions": actions,
        }

        # Process images: uint8 (To, H, W, C) -> float32 (To, C, H, W) in [0, 1]
        for cam_name in self.camera_names:
            if cam_name in sample:
                img = torch.from_numpy(sample[cam_name]).float() / 255.0
                # (To, H, W, C) -> (To, C, H, W)
                img = img.permute(0, 3, 1, 2)
                result[f"{cam_name}_image"] = img

        return result
