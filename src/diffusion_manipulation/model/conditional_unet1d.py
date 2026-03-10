"""ConditionalUnet1D: 1D temporal U-Net for diffusion policy action prediction."""

import torch
import torch.nn as nn

from diffusion_manipulation.model.unet_components import (
    ConditionalResidualBlock1D,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)


class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net with global conditioning for diffusion policy.

    Input: (B, T, action_dim) noisy actions
    Output: (B, T, action_dim) predicted noise

    Conditioning is provided via FiLM (feature-wise linear modulation)
    using a combination of diffusion timestep embedding and global
    observation features.
    """

    def __init__(
        self,
        input_dim: int = 7,
        global_cond_dim: int = 1042,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
        cond_predict_scale: bool = True,
    ) -> None:
        super().__init__()

        # Timestep embedding: scalar -> embed_dim -> 4*embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Total conditioning dim = timestep embed + global observation features
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # Build U-Net levels
        # all_dims: (input_dim, d0, d1, ..., dn)
        # in_out_pairs: [(input_dim, d0), (d0, d1), ..., (d_{n-1}, dn)]
        all_dims = (input_dim,) + tuple(down_dims)
        in_out_pairs = list(zip(all_dims[:-1], all_dims[1:]))

        # Down path: for each (dim_in, dim_out): res1(in→out), res2(out→out), downsample
        self.down_modules = nn.ModuleList()
        for dim_in, dim_out in in_out_pairs:
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out),
            ]))

        # Mid blocks at bottleneck
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        # Up path (reversed): for each reversed (dim_in, dim_out):
        #   upsample(dim_out) → concat skip(dim_out) → res1(2*dim_out → dim_in) → res2(dim_in)
        # After all up levels, x has input_dim channels.
        self.up_modules = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out_pairs):
            self.up_modules.append(nn.ModuleList([
                Upsample1d(dim_out),
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ]))

        # Final 1x1 projection
        self.final_conv = nn.Conv1d(input_dim, input_dim, 1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            sample: (B, T, input_dim) noisy action sequence.
            timestep: (B,) or scalar diffusion timestep.
            global_cond: (B, global_cond_dim) observation features.

        Returns:
            (B, T, input_dim) predicted noise.
        """
        # (B, T, C) -> (B, C, T) for 1D convolutions
        x = sample.permute(0, 2, 1)

        # Encode timestep
        if timestep.dim() == 0:
            timestep = timestep.expand(sample.shape[0])
        timestep_emb = self.diffusion_step_encoder(timestep)  # (B, embed_dim)

        # Combine timestep + global conditioning
        cond = torch.cat([timestep_emb, global_cond], dim=-1)  # (B, cond_dim)

        # Down path with skip connections
        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = downsample(x)

        # Mid
        for mid_block in self.mid_modules:
            x = mid_block(x, cond)

        # Up path: upsample → concat skip → res blocks
        for upsample, res1, res2 in self.up_modules:
            x = upsample(x)
            skip = skips.pop()
            # Handle temporal dimension mismatch from down/upsample rounding
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)

        x = self.final_conv(x)

        # (B, C, T) -> (B, T, C)
        return x.permute(0, 2, 1)
