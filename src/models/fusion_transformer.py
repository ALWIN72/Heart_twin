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
    Dual-stream cross-attention layer: gyro tokens attend to SCG tokens,
    while SCG tokens attend to gyro tokens, with independent stream updates
    and residual MLP feed-forward blocks so modalities remain separate and
    rich across all fusion layers.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.scg_norm1 = nn.LayerNorm(embed_dim)
        self.gyro_norm1 = nn.LayerNorm(embed_dim)
        self.scg_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gyro_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.scg_norm2 = nn.LayerNorm(embed_dim)
        self.gyro_norm2 = nn.LayerNorm(embed_dim)
        self.scg_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.gyro_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, scg_tokens: torch.Tensor, gyro_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cross-attention: SCG attends to Gyro; Gyro attends to SCG
        scg_n, gyro_n = self.scg_norm1(scg_tokens), self.gyro_norm1(gyro_tokens)
        scg_cross, _ = self.scg_attn(query=scg_n, key=gyro_n, value=gyro_n, need_weights=False)
        gyro_cross, _ = self.gyro_attn(query=gyro_n, key=scg_n, value=scg_n, need_weights=False)

        scg_mid = scg_tokens + scg_cross
        gyro_mid = gyro_tokens + gyro_cross

        # Independent stream MLPs
        scg_out = scg_mid + self.scg_mlp(self.scg_norm2(scg_mid))
        gyro_out = gyro_mid + self.gyro_mlp(self.gyro_norm2(gyro_mid))

        return scg_out, gyro_out


class CrossAttentionFusionEncoder(nn.Module):
    """
    Full dual-stream encoder: embed SCG/gyro separately, fuse via cross-
    attention across multiple layers without identity collapse, then project
    back to embed_dim.
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
        for layer in self.fusion_layers:
            scg_tokens, gyro_tokens = layer(scg_tokens, gyro_tokens)

        # Final fusion projection
        fused = torch.cat([scg_tokens, gyro_tokens], dim=-1)
        return self.final_norm(self.proj_back(fused))

