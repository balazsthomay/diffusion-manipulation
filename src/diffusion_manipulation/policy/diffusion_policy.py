"""Diffusion policy using ConditionalUnet1D for action prediction."""

import torch
import torch.nn as nn

from diffusion_manipulation.data.normalizer import LinearNormalizer
from diffusion_manipulation.model.conditional_unet1d import ConditionalUnet1D
from diffusion_manipulation.model.noise_schedulers import create_noise_scheduler
from diffusion_manipulation.model.vision_encoder import VisionEncoder
from diffusion_manipulation.policy.base_policy import BasePolicy


class DiffusionUnetPolicy(BasePolicy):
    """Diffusion policy with DDPM training and DDIM inference.

    Training: predicts noise ε added to actions at random timesteps.
    Inference: iteratively denoises random noise into action sequences.
    """

    def __init__(
        self,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        lowdim_obs_dim: int = 9,
        n_diffusion_steps_train: int = 100,
        n_diffusion_steps_infer: int = 16,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
        cond_predict_scale: bool = True,
        vision_feature_dim: int = 512,
        crop_shape: tuple[int, int] = (76, 76),
        pretrained_vision: bool = True,
        camera_names: tuple[str, ...] = ("agentview",),
        beta_schedule: str = "squaredcos_cap_v2",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.camera_names = camera_names

        # Vision encoder
        self.vision_encoder = VisionEncoder(
            input_shape=(84, 84),
            crop_shape=crop_shape,
            feature_dim=vision_feature_dim,
            pretrained=pretrained_vision,
            num_cameras=len(camera_names),
        )

        # Calculate global conditioning dimension
        global_cond_dim = self.vision_encoder.get_output_dim(
            obs_horizon=obs_horizon,
            lowdim_dim=lowdim_obs_dim,
            num_cameras=len(camera_names),
        )

        # U-Net noise prediction network
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            cond_predict_scale=cond_predict_scale,
        )

        # Noise schedulers
        self.noise_scheduler_train = create_noise_scheduler(
            scheduler_type="ddpm",
            num_train_timesteps=n_diffusion_steps_train,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        self.noise_scheduler_infer = create_noise_scheduler(
            scheduler_type="ddim",
            num_train_timesteps=n_diffusion_steps_train,
            num_inference_steps=n_diffusion_steps_infer,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )

        self.normalizer: LinearNormalizer | None = None
        self.n_diffusion_steps_train = n_diffusion_steps_train

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = normalizer

    def _encode_obs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode observations into a global conditioning vector."""
        images = {}
        for cam_name in self.camera_names:
            key = f"{cam_name}_image"
            if key in batch:
                images[cam_name] = batch[key]

        lowdim = batch["lowdim_obs"]
        return self.vision_encoder(images, lowdim)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute DDPM training loss.

        1. Encode observations → global_cond
        2. Sample random noise ε and timestep t
        3. Create noisy actions: x_t = scheduler.add_noise(actions, ε, t)
        4. Predict noise: ε_pred = unet(x_t, t, global_cond)
        5. Return MSE(ε_pred, ε)
        """
        # Encode observations
        global_cond = self._encode_obs(batch)

        # Get normalized actions
        actions = batch["actions"]  # (B, Tp, action_dim)
        B = actions.shape[0]

        # Sample noise and timestep
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self.n_diffusion_steps_train, (B,),
            device=actions.device,
        )

        # Add noise to actions
        noisy_actions = self.noise_scheduler_train.add_noise(actions, noise, timesteps)

        # Predict noise
        noise_pred = self.unet(noisy_actions, timesteps, global_cond)

        # MSE loss
        return nn.functional.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict action sequence via DDIM denoising.

        1. Encode observations → global_cond
        2. Initialize random noise x_T ~ N(0, I)
        3. Iteratively denoise: x_{t-1} = DDIM_step(x_t, t, unet(x_t, t, cond))
        4. Unnormalize and return action_horizon slice

        Args:
            obs: Dict with observation tensors. Each should have batch dim.

        Returns:
            (B, action_horizon, action_dim) unnormalized action tensor.
        """
        self.eval()
        device = next(self.parameters()).device

        # Ensure batch dimension
        has_batch = obs["lowdim_obs"].dim() == 3
        if not has_batch:
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}

        # Move to device
        obs = {k: v.to(device) for k, v in obs.items()}

        # Encode observations
        global_cond = self._encode_obs(obs)
        B = global_cond.shape[0]

        # Initialize from noise
        noisy_actions = torch.randn(
            (B, self.pred_horizon, self.action_dim),
            device=device,
        )

        # DDIM denoising loop
        self.noise_scheduler_infer.set_timesteps(len(self.noise_scheduler_infer.timesteps))
        for t in self.noise_scheduler_infer.timesteps:
            noise_pred = self.unet(
                noisy_actions,
                t.to(device),
                global_cond,
            )
            noisy_actions = self.noise_scheduler_infer.step(
                model_output=noise_pred,
                timestep=int(t),
                sample=noisy_actions,
            ).prev_sample

        # Unnormalize actions
        if self.normalizer is not None:
            noisy_actions = self.normalizer.unnormalize("actions", noisy_actions)

        # Extract action_horizon window (offset by obs_horizon - 1)
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        actions = noisy_actions[:, start:end, :]

        if not has_batch:
            actions = actions.squeeze(0)

        return actions
