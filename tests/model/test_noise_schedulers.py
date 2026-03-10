"""Tests for noise scheduler factory."""

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from diffusion_manipulation.model.noise_schedulers import create_noise_scheduler


class TestCreateNoiseScheduler:
    def test_ddpm_type(self) -> None:
        sched = create_noise_scheduler(scheduler_type="ddpm")
        assert isinstance(sched, DDPMScheduler)

    def test_ddim_type(self) -> None:
        sched = create_noise_scheduler(scheduler_type="ddim")
        assert isinstance(sched, DDIMScheduler)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_noise_scheduler(scheduler_type="invalid")

    def test_ddpm_config(self) -> None:
        sched = create_noise_scheduler(
            scheduler_type="ddpm",
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        assert sched.config.num_train_timesteps == 100
        assert sched.config.beta_schedule == "squaredcos_cap_v2"
        assert sched.config.prediction_type == "epsilon"

    def test_ddim_inference_steps(self) -> None:
        sched = create_noise_scheduler(
            scheduler_type="ddim",
            num_train_timesteps=100,
            num_inference_steps=16,
        )
        assert len(sched.timesteps) == 16

    def test_add_noise_roundtrip(self) -> None:
        sched = create_noise_scheduler(scheduler_type="ddpm", num_train_timesteps=100)
        original = torch.randn(2, 16, 7)
        noise = torch.randn_like(original)
        timesteps = torch.tensor([0, 0])

        noisy = sched.add_noise(original, noise, timesteps)
        # At t=0, noise should be minimal
        assert noisy.shape == original.shape

    def test_clip_sample_config(self) -> None:
        sched = create_noise_scheduler(
            scheduler_type="ddpm",
            clip_sample=True,
            clip_sample_range=1.0,
        )
        assert sched.config.clip_sample is True
        assert sched.config.clip_sample_range == 1.0
