"""
src/models/fusion_transformer.py

Cross-modal fusion components. The 6-channel input is already fused at the
patch-embedding level inside SCGEncoder (see models/encoder.py), but for
tasks where we want to reason about the SCG and gyro/gyro-proxy streams as
*separate* modalities that attend to each other (e.g. motion-artifact
cancellation, where gyro should actively explain away SCG noise), we split
the 6 channels into two 3-channel streams and fuse them via cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.encoder import FusionPatchEmbed1D, SinusoidalPositionalEncoding


class DualStreamPatchEmbed(nn.Module):
    """Separately patch-embeds the SCG (first 3 channels) and gyro (last 3) streams."""

    def __init__(self, patch_size: int = 125, embed_dim: int = 256):
        super().__init__()
        self.scg_embed = FusionPatchEmbed1D(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)
        self.gyro_embed = FusionPatchEmbed1D(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)
        self.pos = SinusoidalPositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 6, T) -> (scg_tokens, gyro_tokens), each (B, num_patches, embed_dim)."""
        scg, gyro = x[:, :3, :], x[:, 3:, :]
        scg_tokens = self.pos(self.scg_embed(scg))
        gyro_tokens = self.pos(self.gyro_embed(gyro))
        return scg_tokens, gyro_tokens


class CrossModalFusion(nn.Module):
    """
    Breakthrough fusion layer: gyro tokens attend to SCG tokens to enhance
    the cardiac signal, while SCG tokens attend to gyro tokens to identify
    and cancel motion artifact.

    Returns the concatenated, residual-connected fused tokens.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.scg_norm = nn.LayerNorm(embed_dim)
        self.gyro_norm = nn.LayerNorm(embed_dim)
        self.scg_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gyro_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim * 2)

    def forward(self, scg_tokens: torch.Tensor, gyro_tokens: torch.Tensor) -> torch.Tensor:
        scg_n, gyro_n = self.scg_norm(scg_tokens), self.gyro_norm(gyro_tokens)

        # SCG attends to Gyro: "where does motion explain my variance?"
        scg_attend_gyro, _ = self.scg_attn(query=scg_n, key=gyro_n, value=gyro_n, need_weights=False)
        scg_fused = scg_tokens + scg_attend_gyro

        # Gyro attends to SCG: "which of my patterns correlate with cardiac timing?"
        gyro_attend_scg, _ = self.gyro_attn(query=gyro_n, key=scg_n, value=scg_n, need_weights=False)
        gyro_fused = gyro_tokens + gyro_attend_scg

        fused = torch.cat([scg_fused, gyro_fused], dim=-1)  # (B, num_patches, 2*embed_dim)
        return self.out_proj(fused)


class CrossAttentionFusionEncoder(nn.Module):
    """
    Full dual-stream encoder: embed SCG/gyro separately, fuse via cross-
    attention, then project back to embed_dim. Used by the denoiser and as
    an optional richer alternative to the single-stream SCGEncoder for
    tasks that benefit from explicit modality separation before fusion.
    """

    def __init__(self, patch_size: int = 125, embed_dim: int = 256, num_heads: int = 8,
                 fusion_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dual_embed = DualStreamPatchEmbed(patch_size, embed_dim)
        self.fusion_layers = nn.ModuleList([
            CrossModalFusion(embed_dim, num_heads, dropout) for _ in range(fusion_layers)
        ])
        self.proj_back = nn.Linear(embed_dim * 2, embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, T) -> (B, num_patches, embed_dim) fused tokens."""
        scg_tokens, gyro_tokens = self.dual_embed(x)
        fused_2d = None
        for layer in self.fusion_layers:
            fused_2d = layer(scg_tokens, gyro_tokens)             # (B, P, 2*embed_dim)
            half = self.proj_back(fused_2d)                       # (B, P, embed_dim)
            scg_tokens, gyro_tokens = half, half                  # feed forward to next fusion layer
        return self.final_norm(self.proj_back(fused_2d))
