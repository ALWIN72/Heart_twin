"""
src/models/denoiser.py

Phase 2: Motion Artifact Cancellation.

Uses the gyro/gyro-proxy stream to actively explain away and subtract
motion artifact from the SCG stream. Built on top of the frozen (early
layers) pretrained foundation encoder, with a trainable cancellation head
that reconstructs a clean per-patch SCG waveform from fused SCG+gyro
features.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.encoder import SCGEncoder
from src.models.fusion_transformer import CrossAttentionFusionEncoder


class MotionArtifactCanceller(nn.Module):
    """
    Wraps a (typically pretrained) foundation encoder + a dedicated
    cross-attention fusion encoder, and predicts a clean SCG waveform per
    patch. Trained with paired (noisy, clean) windows -- in practice,
    "clean" targets are calibrated SCG segments with low gyro-proxy energy
    (i.e. naturally still recordings), and "noisy" inputs are the same
    segments with synthetic motion bursts injected (see
    train_artifact_cancellation in finetune_denoiser.py).
    """

    def __init__(self, foundation_encoder: SCGEncoder, cfg: ModelConfig | None = None,
                 freeze_early_layers: int = 2):
        super().__init__()
        self.cfg = cfg = cfg or foundation_encoder.cfg
        self.encoder = foundation_encoder

        # Freeze the first `freeze_early_layers` transformer blocks (low-level
        # features); fine-tune the rest. Freezing "the first 10 parameters"
        # as in the original sketch doesn't correspond to meaningful layer
        # boundaries, so we freeze whole blocks instead.
        for i, block in enumerate(self.encoder.blocks):
            if i < freeze_early_layers:
                for p in block.parameters():
                    p.requires_grad = False
        for p in self.encoder.patch_embed.parameters():
            p.requires_grad = False

        # Dedicated fusion pathway that explicitly cross-attends SCG <-> gyro
        # (separate from the foundation encoder's early-fused single stream).
        self.fusion_encoder = CrossAttentionFusionEncoder(
            patch_size=cfg.patch_size, embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
            fusion_layers=2, dropout=cfg.dropout,
        )

        self.cancellation_head = nn.Sequential(
            nn.Linear(cfg.embed_dim * 2, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, 3 * cfg.patch_size),  # reconstruct clean 3-axis SCG patch
        )

    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        """
        x_noisy: (B, 6, T) noisy 6-channel window (SCG possibly motion-corrupted).
        Returns: (B, N, 3*patch_size) predicted clean SCG patches (flattened per-patch x,y,z).
        """
        foundation_tokens = self.encoder(x_noisy)          # (B, N, embed_dim) -- frozen/semi-frozen global context
        fusion_tokens = self.fusion_encoder(x_noisy)        # (B, N, embed_dim) -- explicit SCG<->gyro cross-attention
        combined = torch.cat([foundation_tokens, fusion_tokens], dim=-1)  # (B, N, 2*embed_dim)
        clean_patches = self.cancellation_head(combined)    # (B, N, 3*patch_size)
        return clean_patches

    def reconstruct_waveform(self, x_noisy: torch.Tensor) -> torch.Tensor:
        """Convenience: returns the predicted clean SCG as a (B, 3, T) waveform instead of flattened patches."""
        patches = self.forward(x_noisy)  # (B, N, 3*P)
        b, n, _ = patches.shape
        p = self.cfg.patch_size
        patches = patches.view(b, n, 3, p).permute(0, 2, 1, 3)  # (B, 3, N, P)
        return patches.reshape(b, 3, n * p)
