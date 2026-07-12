"""
src/models/encoder.py

Core transformer encoder backbone shared by every downstream model
(denoiser, harmonizer, placement classifier, personal twin). This is the
"foundation model" referenced throughout the roadmap: pretrained once via
masked-autoencoding (src/models/mae.py), then frozen or fine-tuned for
every downstream task.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.config import ModelConfig


class FusionPatchEmbed1D(nn.Module):
    """
    Splits a (B, C, T) multi-channel signal into non-overlapping patches
    along time and projects each patch to embed_dim. With C=6 (3 SCG + 3
    gyro/gyro-proxy axes), this jointly embeds both modalities from the
    first layer, rather than embedding them separately -- the early fusion
    keeps the patch count (and therefore attention cost) low while still
    letting the encoder learn cross-axis structure immediately.
    """

    def __init__(self, in_channels: int = 6, patch_size: int = 125, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding over the patch sequence."""

    def __init__(self, embed_dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block (MHSA + MLP, residual connections)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SCGEncoder(nn.Module):
    """
    The foundation-model backbone: patch-embed the 6-channel input, add
    positional encoding, and run through a stack of transformer blocks.

    forward(x) -> (B, num_patches, embed_dim) token sequence. Downstream
    heads typically mean-pool this over the patch dimension to get a single
    (B, embed_dim) global feature vector (see e.g. PersonalHeartTwin).
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg

        self.patch_embed = FusionPatchEmbed1D(cfg.in_channels, cfg.patch_size, cfg.embed_dim)
        self.pos_embed = SinusoidalPositionalEncoding(cfg.embed_dim, max_len=cfg.num_patches + 8)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.encoder_depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T) -> tokens: (B, num_patches, embed_dim)"""
        tokens = self.patch_embed(x)
        tokens = self.pos_embed(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    def encode_pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: mean-pooled global feature vector, (B, embed_dim)."""
        return self.forward(x).mean(dim=1)
