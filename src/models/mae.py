"""
src/models/mae.py

Masked Autoencoder (MAE) pretraining objective for the SCG foundation
model. We randomly mask a high fraction (default 75%) of the patches in
each window, encode only the visible patches, then ask a lightweight
decoder to reconstruct the full (raw, normalized) signal for every patch,
visible and masked. This is the self-supervised pretext task that gives
the foundation model (SCGEncoder) reusable cardiac-vibration features
without needing any labels.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.encoder import SCGEncoder, SinusoidalPositionalEncoding, TransformerBlock


def random_masking(tokens: torch.Tensor, mask_ratio: float):
    """
    Per-sample random masking via shuffling, the standard MAE trick.

    tokens: (B, N, D)
    Returns:
        kept_tokens: (B, N_keep, D) -- tokens that remain visible to the encoder
        mask: (B, N) binary mask, 1 = masked/removed, 0 = kept
        ids_restore: (B, N) indices to unshuffle back to original patch order
    """
    b, n, d = tokens.shape
    n_keep = max(1, int(n * (1 - mask_ratio)))

    noise = torch.rand(b, n, device=tokens.device)
    ids_shuffle = torch.argsort(noise, dim=1)          # ascending: small noise = keep first
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :n_keep]
    kept_tokens = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

    mask = torch.ones(b, n, device=tokens.device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # unshuffle mask to original order

    return kept_tokens, mask, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight decoder: reconstructs raw per-patch signal values from latent tokens."""

    def __init__(self, encoder_embed_dim: int, decoder_embed_dim: int, decoder_depth: int,
                 num_heads: int, num_patches: int, patch_size: int, out_channels: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.pos_embed = SinusoidalPositionalEncoding(decoder_embed_dim, max_len=num_patches + 8)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, out_channels * patch_size)
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(self, kept_latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        kept_latent: (B, N_keep, encoder_embed_dim) -- encoder output for visible patches
        ids_restore: (B, N) -- to put mask tokens + visible tokens back in original order
        Returns: (B, N, out_channels * patch_size) reconstructed patches (flattened)
        """
        x = self.decoder_embed(kept_latent)  # (B, N_keep, decoder_embed_dim)
        b, n_keep, d = x.shape
        n = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(b, n - n_keep, 1)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, D), kept first then masked
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d))

        x_full = self.pos_embed(x_full)
        for block in self.blocks:
            x_full = block(x_full)
        x_full = self.norm(x_full)
        return self.pred(x_full)  # (B, N, C*patch_size)


class MaskedAutoencoder(nn.Module):
    """
    Full MAE: SCGEncoder (patch-embed + mask + transformer) -> MAEDecoder.
    Loss is MSE on masked patches only (standard MAE choice -- reconstructing
    visible patches is trivial and provides no learning signal).
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.mask_ratio = cfg.mask_ratio

        # Re-use encoder's patch embed + pos embed, but we need to intercept
        # *after* patch embedding to apply masking before the transformer
        # blocks, so we don't reuse SCGEncoder.forward() directly.
        self.patch_embed = SCGEncoder(cfg).patch_embed
        self.pos_embed = SinusoidalPositionalEncoding(cfg.embed_dim, max_len=cfg.num_patches + 8)
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        self.decoder = MAEDecoder(
            encoder_embed_dim=cfg.embed_dim,
            decoder_embed_dim=cfg.decoder_embed_dim,
            decoder_depth=cfg.decoder_depth,
            num_heads=cfg.num_heads,
            num_patches=cfg.num_patches,
            patch_size=cfg.patch_size,
            out_channels=cfg.in_channels,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    def patchify_target(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, N, C*patch_size) ground-truth patches, matching decoder output layout."""
        b, c, t = x.shape
        p = self.cfg.patch_size
        n = t // p
        x = x[:, :, : n * p].reshape(b, c, n, p)        # (B, C, N, P)
        x = x.permute(0, 2, 1, 3).reshape(b, n, c * p)  # (B, N, C*P)
        return x

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T) raw (normalized) multi-channel window.
        Returns dict with loss, predicted patches, mask, and target patches
        (predictions/targets are handy for visualization/debugging).
        """
        tokens = self.patch_embed(x)              # (B, N, D)
        tokens = self.pos_embed(tokens)

        kept_tokens, mask, ids_restore = random_masking(tokens, self.mask_ratio)

        latent = kept_tokens
        for block in self.encoder_blocks:
            latent = block(latent)
        latent = self.encoder_norm(latent)

        pred = self.decoder(latent, ids_restore)   # (B, N, C*P)
        target = self.patchify_target(x)            # (B, N, C*P)

        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp(min=1)

        return {"loss": loss, "pred": pred, "target": target, "mask": mask}

    def export_encoder(self) -> SCGEncoder:
        """Build a standalone SCGEncoder and load this MAE's pretrained encoder weights into it."""
        encoder = SCGEncoder(self.cfg)
        encoder.patch_embed.load_state_dict(self.patch_embed.state_dict())
        encoder.pos_embed.load_state_dict(self.pos_embed.state_dict())
        encoder.blocks.load_state_dict(self.encoder_blocks.state_dict())
        encoder.norm.load_state_dict(self.encoder_norm.state_dict())
        return encoder
