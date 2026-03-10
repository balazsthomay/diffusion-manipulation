"""Building blocks for the ConditionalUnet1D architecture."""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps scalar timestep (B,) to embedding vector (B, dim).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x.unsqueeze(-1).float() * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """Downsample temporal dimension by 2 using strided convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsample temporal dimension by 2 using transposed convolution."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning.

    Two Conv1dBlocks with a conditioning projection applied after the
    first convolution (scale and bias via FiLM).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()
        self.cond_predict_scale = cond_predict_scale

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])

        # FiLM conditioning: project cond to scale + bias (or just bias)
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )

        # Residual connection with optional projection
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input features.
            cond: (B, cond_dim) conditioning vector.

        Returns:
            (B, out_channels, T) output features.
        """
        out = self.blocks[0](x)

        # Apply FiLM conditioning
        cond_embed = self.cond_encoder(cond)  # (B, cond_channels)
        cond_embed = cond_embed.unsqueeze(-1)  # (B, cond_channels, 1)

        if self.cond_predict_scale:
            scale, bias = cond_embed.chunk(2, dim=1)
            out = out * (1 + scale) + bias
        else:
            out = out + cond_embed

        out = self.blocks[1](out)
        return out + self.residual_conv(x)
