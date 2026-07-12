"""
src/models/placement_classifier.py

Phase 3.5.1: Placement Detection.

A small 1D-CNN that uses the raw 6-channel (SCG + gyro/gyro-proxy) window
to classify phone placement on the chest: Sternum, Left, or Right.
Lightweight by design -- this needs to run cheaply and frequently
(potentially every new recording) on-device or in a fast preprocessing step.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.config import N_CHANNELS, PLACEMENT_CLASSES, WINDOW_SAMPLES


class PlacementClassifier(nn.Module):
    """
    Uses raw 6-channel amplitude/orientation patterns to detect placement.

    The original spec hard-coded the flatten size (64*10) assuming a fixed
    input length; that breaks for any window length other than the one it
    was written against. We instead use adaptive global average pooling so
    the classifier works for any input length >= a few conv strides.
    """

    def __init__(self, in_channels: int = N_CHANNELS, num_classes: int = len(PLACEMENT_CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, T) -> (B, num_classes) logits."""
        feat = self.conv(x)
        feat = self.pool(feat).squeeze(-1)  # (B, 128)
        return self.fc(feat)

    @torch.no_grad()
    def predict_label(self, x: torch.Tensor) -> list[str]:
        logits = self.forward(x)
        idx = logits.argmax(dim=-1).tolist()
        return [PLACEMENT_CLASSES[i] for i in idx]
