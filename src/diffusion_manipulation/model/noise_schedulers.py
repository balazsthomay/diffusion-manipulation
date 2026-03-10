"""Noise scheduler factory wrapping diffusers schedulers."""

from diffusers import DDIMScheduler, DDPMScheduler


def create_noise_scheduler(
    scheduler_type: str = "ddpm",
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    num_inference_steps: int | None = None,
) -> DDPMScheduler | DDIMScheduler:
    """Create a noise scheduler for training or inference.

    Args:
        scheduler_type: "ddpm" for training, "ddim" for inference.
        num_train_timesteps: Number of diffusion steps during training.
        beta_schedule: Beta schedule type (e.g., "squaredcos_cap_v2").
        prediction_type: What the model predicts ("epsilon" or "sample").
        clip_sample: Whether to clip predicted samples.
        clip_sample_range: Range for sample clipping.
        num_inference_steps: Steps for DDIM inference (ignored for DDPM).

    Returns:
        Configured scheduler instance.
    """
    if scheduler_type == "ddpm":
        scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )
        if num_inference_steps is not None:
            scheduler.set_timesteps(num_inference_steps)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'ddpm' or 'ddim'.")

    return scheduler
