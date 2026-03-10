"""Tests for ConditionalUnet1D."""

import torch

from diffusion_manipulation.model.conditional_unet1d import ConditionalUnet1D


class TestConditionalUnet1D:
    """Tests use tiny config (down_dims=(8,16), n_groups=2) for speed."""

    def _make_tiny_unet(self, **kwargs) -> ConditionalUnet1D:
        defaults = dict(
            input_dim=4,
            global_cond_dim=16,
            down_dims=(8, 16),
            kernel_size=3,
            n_groups=2,
            diffusion_step_embed_dim=8,
        )
        defaults.update(kwargs)
        return ConditionalUnet1D(**defaults)

    def test_output_shape(self) -> None:
        unet = self._make_tiny_unet()
        B, T, D = 2, 16, 4
        sample = torch.randn(B, T, D)
        timestep = torch.randint(0, 100, (B,))
        global_cond = torch.randn(B, 16)

        out = unet(sample, timestep, global_cond)
        assert out.shape == (B, T, D)

    def test_scalar_timestep(self) -> None:
        unet = self._make_tiny_unet()
        sample = torch.randn(2, 16, 4)
        timestep = torch.tensor(50)
        global_cond = torch.randn(2, 16)

        out = unet(sample, timestep, global_cond)
        assert out.shape == (2, 16, 4)

    def test_gradient_flow(self) -> None:
        unet = self._make_tiny_unet()
        sample = torch.randn(2, 16, 4, requires_grad=True)
        timestep = torch.randint(0, 100, (2,))
        global_cond = torch.randn(2, 16, requires_grad=True)

        out = unet(sample, timestep, global_cond)
        loss = out.sum()
        loss.backward()

        assert sample.grad is not None
        assert global_cond.grad is not None

    def test_different_timesteps_give_different_outputs(self) -> None:
        unet = self._make_tiny_unet()
        unet.eval()

        sample = torch.randn(1, 16, 4)
        global_cond = torch.randn(1, 16)

        with torch.no_grad():
            out1 = unet(sample, torch.tensor([0]), global_cond)
            out2 = unet(sample, torch.tensor([99]), global_cond)

        assert not torch.allclose(out1, out2)

    def test_deterministic_eval(self) -> None:
        unet = self._make_tiny_unet()
        unet.eval()

        sample = torch.randn(1, 16, 4)
        timestep = torch.tensor([50])
        global_cond = torch.randn(1, 16)

        with torch.no_grad():
            out1 = unet(sample, timestep, global_cond)
            out2 = unet(sample, timestep, global_cond)

        assert torch.equal(out1, out2)

    def test_various_temporal_lengths(self) -> None:
        """U-Net should handle various temporal dimensions."""
        unet = self._make_tiny_unet()

        for T in [8, 16, 32]:
            sample = torch.randn(1, T, 4)
            timestep = torch.tensor([10])
            global_cond = torch.randn(1, 16)

            out = unet(sample, timestep, global_cond)
            assert out.shape == (1, T, 4), f"Failed for T={T}"
