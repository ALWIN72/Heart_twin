"""
src/models/dann.py

Phase 3: Phone-Independent Calibration via Domain-Adversarial Training.

Goal: make the foundation encoder's features invariant to which phone
recorded the signal (iOS vs Android, or even specific device models),
while still being able to reconstruct the original signal. The gradient
reversal layer flips the sign of gradients flowing from a device
classifier back into the encoder, so the encoder is explicitly penalized
for encoding device-identifying information -- the classic DANN trick.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import ModelConfig
from src.models.encoder import SCGEncoder


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalLayer.apply(x, alpha)


class DeviceHarmonizer(nn.Module):
    """
    Wraps a foundation encoder with two heads:
      - reconstruction_head: rebuilds the per-patch signal from device-
        invariant features (keeps the representation informative, not just
        adversarially collapsed).
      - domain_classifier: predicts which device/platform produced the
        recording, fed through a gradient-reversal layer so that training
        it to be *accurate* simultaneously trains the encoder to make its
        features *useless* for that prediction.
    """

    def __init__(self, encoder: SCGEncoder, num_devices: int = 2, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or encoder.cfg
        self.encoder = encoder
        self.reconstruction_head = nn.Linear(cfg.embed_dim, cfg.patch_size * cfg.in_channels)
        self.domain_classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, 64), nn.ReLU(), nn.Linear(64, num_devices)
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """
        x: (B, in_channels, T)
        Returns:
            recon: (B, N, in_channels*patch_size) per-patch reconstruction
            domain_pred: (B, num_devices) logits (gradient-reversed into encoder)
        """
        encoded = self.encoder(x)            # (B, N, embed_dim)
        pooled = encoded.mean(dim=1)         # (B, embed_dim)

        recon = self.reconstruction_head(encoded)  # device-agnostic per-patch reconstruction

        reversed_feat = grad_reverse(pooled, alpha)
        domain_pred = self.domain_classifier(reversed_feat)

        return recon, domain_pred

    @torch.no_grad()
    def harmonize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-time use: run a recording through the device-invariant
        encoder + reconstruction head to produce a "virtual canonical
        device" version of the signal, suitable for comparing recordings
        made on different phones.
        """
        encoded = self.encoder(x)
        recon_patches = self.reconstruction_head(encoded)   # (B, N, C*P)
        b, n, _ = recon_patches.shape
        c, p = self.cfg.in_channels, self.cfg.patch_size
        recon_patches = recon_patches.view(b, n, c, p).permute(0, 2, 1, 3)
        return recon_patches.reshape(b, c, n * p)
