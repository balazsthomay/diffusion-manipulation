"""Robosuite environment wrapper for standardized observation and action interface."""

from collections import deque

import numpy as np
import numpy.typing as npt


class RobosuiteEnv:
    """Wrapper around robosuite environments.

    Provides standardized reset/step/get_obs interface with
    consistent observation extraction and action space handling.
    """

    def __init__(
        self,
        task_name: str = "Lift",
        robots: str = "Panda",
        camera_names: tuple[str, ...] = ("agentview",),
        camera_height: int = 84,
        camera_width: int = 84,
        control_freq: int = 20,
        horizon: int = 400,
        use_camera_obs: bool = True,
        seed: int | None = None,
    ) -> None:
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller="BASIC",
            robot="Panda",
        )

        self.camera_names = camera_names
        self.env = suite.make(
            env_name=task_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=use_camera_obs,
            use_camera_obs=use_camera_obs,
            camera_names=list(camera_names),
            camera_heights=camera_height,
            camera_widths=camera_width,
            control_freq=control_freq,
            horizon=horizon,
        )

        if seed is not None:
            self.env.seed(seed)

        self._obs: dict | None = None

    @property
    def action_dim(self) -> int:
        return self.env.action_spec[0].shape[0]

    def reset(self) -> dict[str, npt.NDArray]:
        """Reset environment and return initial observation."""
        self._obs = self.env.reset()
        return self.get_obs()

    def step(self, action: npt.NDArray[np.float32]) -> tuple[dict[str, npt.NDArray], float, bool, dict]:
        """Take a step in the environment.

        Args:
            action: (action_dim,) action array.

        Returns:
            Tuple of (obs, reward, done, info).
        """
        self._obs, reward, done, info = self.env.step(action)
        return self.get_obs(), reward, done, info

    def get_obs(self) -> dict[str, npt.NDArray]:
        """Extract standardized observations from raw env obs.

        Returns:
            Dict with:
                - 'lowdim': (D,) array with eef_pos, eef_quat, gripper_qpos
                - '{camera_name}_image': (H, W, C) uint8 array for each camera
        """
        obs = self._obs
        lowdim_parts = []

        if "robot0_eef_pos" in obs:
            lowdim_parts.append(obs["robot0_eef_pos"].astype(np.float32))
        if "robot0_eef_quat" in obs:
            lowdim_parts.append(obs["robot0_eef_quat"].astype(np.float32))
        if "robot0_gripper_qpos" in obs:
            lowdim_parts.append(obs["robot0_gripper_qpos"].astype(np.float32))

        result: dict[str, npt.NDArray] = {
            "lowdim": np.concatenate(lowdim_parts) if lowdim_parts else np.array([], dtype=np.float32),
        }

        for cam_name in self.camera_names:
            img_key = f"{cam_name}_image"
            if img_key in obs:
                result[img_key] = obs[img_key].astype(np.uint8)

        return result

    def close(self) -> None:
        self.env.close()

    def check_success(self) -> bool:
        """Check if the current state is a success."""
        return bool(self.env._check_success())
