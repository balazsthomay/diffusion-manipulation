"""Tests for U-Net building blocks."""

import torch
import pytest

from diffusion_manipulation.model.unet_components import (
    SinusoidalPosEmb,
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
    ConditionalResidualBlock1D,
)


class TestSinusoidalPosEmb:
    def test_output_shape(self) -> None:
        emb = SinusoidalPosEmb(dim=64)
        x = torch.tensor([0, 1, 50, 99], dtype=torch.float32)
        out = emb(x)
        assert out.shape == (4, 64)

    def test_different_timesteps_differ(self) -> None:
        emb = SinusoidalPosEmb(dim=32)
        t1 = emb(torch.tensor([0.0]))
        t2 = emb(torch.tensor([50.0]))
        assert not torch.allclose(t1, t2)

    def test_deterministic(self) -> None:
        emb = SinusoidalPosEmb(dim=32)
        x = torch.tensor([10.0, 20.0])
        assert torch.equal(emb(x), emb(x))


class TestConv1dBlock:
    def test_output_shape(self) -> None:
        block = Conv1dBlock(in_channels=8, out_channels=16, kernel_size=5, n_groups=4)
        x = torch.randn(2, 8, 16)  # (B, C, T)
        out = block(x)
        assert out.shape == (2, 16, 16)

    def test_preserves_temporal_dim(self) -> None:
        block = Conv1dBlock(in_channels=4, out_channels=8, kernel_size=3, n_groups=2)
        x = torch.randn(1, 4, 32)
        assert block(x).shape[-1] == 32


class TestDownsample1d:
    def test_halves_temporal(self) -> None:
        ds = Downsample1d(dim=8)
        x = torch.randn(2, 8, 16)
        out = ds(x)
        assert out.shape == (2, 8, 8)


class TestUpsample1d:
    def test_doubles_temporal(self) -> None:
        us = Upsample1d(dim=8)
        x = torch.randn(2, 8, 8)
        out = us(x)
        assert out.shape == (2, 8, 16)


class TestConditionalResidualBlock1D:
    def test_output_shape_same_channels(self) -> None:
        block = ConditionalResidualBlock1D(
            in_channels=8, out_channels=8, cond_dim=16, kernel_size=3, n_groups=2
        )
        x = torch.randn(2, 8, 16)
        cond = torch.randn(2, 16)
        out = block(x, cond)
        assert out.shape == (2, 8, 16)

    def test_output_shape_different_channels(self) -> None:
        block = ConditionalResidualBlock1D(
            in_channels=8, out_channels=16, cond_dim=32, kernel_size=3, n_groups=2
        )
        x = torch.randn(2, 8, 16)
        cond = torch.randn(2, 32)
        out = block(x, cond)
        assert out.shape == (2, 16, 16)

    def test_gradient_flow(self) -> None:
        block = ConditionalResidualBlock1D(
            in_channels=4, out_channels=8, cond_dim=16, kernel_size=3, n_groups=2
        )
        x = torch.randn(2, 4, 16, requires_grad=True)
        cond = torch.randn(2, 16, requires_grad=True)
        out = block(x, cond)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert cond.grad is not None

    def test_without_scale(self) -> None:
        block = ConditionalResidualBlock1D(
            in_channels=8, out_channels=8, cond_dim=16,
            kernel_size=3, n_groups=2, cond_predict_scale=False,
        )
        x = torch.randn(2, 8, 16)
        cond = torch.randn(2, 16)
        out = block(x, cond)
        assert out.shape == (2, 8, 16)
