"""
src/models/spatial_transformer.py

Phase 3.5.2: Spatial Correction.

Goal: if the phone is on the Left or Right chest instead of the Sternum,
rotate/mix the 3 accelerometer axes (and 3 gyro axes) to approximate what
the signal would have looked like from the Sternum position.

Why not torch's affine_grid/grid_sample (as in the original sketch)?
`affine_grid` + `grid_sample` are built for spatially resampling 2D/3D
*images* -- warping pixel coordinates. Our data is a 1D multi-channel time
series; there is no "spatial grid" to resample, and feeding (2,3) affine
parameters into grid_sample on a (B, 6, 2500) tensor doesn't correspond to
any physically meaningful operation. The actual physical correction we
want is a per-placement *linear mixing of the channel axes* (the gyro
+ accelerometer triplets rotate together when you move the phone from
sternum to left/right chest) -- i.e. a small rotation matrix applied at
every timestep, not a spatial resampling. We implement that directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import PLACEMENT_CLASSES


class SpatialCorrector(nn.Module):
    """
    Learns a (3x3) rotation-like matrix per placement class for the SCG
    axes, and a separate one for the gyro axes, then applies it as a
    per-timestep linear transform: x_corrected[:, t] = R @ x[:, t].

    Placement is supplied either as soft class probabilities (e.g. softmax
    output of PlacementClassifier) or hard class indices; soft probabilities
    let gradients flow end-to-end if this is ever trained jointly with the
    classifier.
    """

    def __init__(self, num_classes: int = len(PLACEMENT_CLASSES)):
        super().__init__()
        self.num_classes = num_classes
        # One 3x3 matrix per class per modality (SCG, gyro). Initialized
        # near identity so an untrained model starts as a no-op correction.
        init = torch.eye(3).unsqueeze(0).repeat(num_classes, 1, 1)
        init = init + 0.01 * torch.randn_like(init)
        self.scg_rotations = nn.Parameter(init.clone())
        self.gyro_rotations = nn.Parameter(init.clone())

    def forward(self, x: torch.Tensor, placement_probs: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, T) -- channels 0:3 = SCG axes, 3:6 = gyro axes
        placement_probs: (B, num_classes) soft or one-hot placement
            probabilities (e.g. softmax(placement_logits) from PlacementClassifier)
        Returns: (B, 6, T) corrected signal.
        """
        # Mix per-class rotation matrices by the placement probabilities ->
        # one effective rotation matrix per sample.
        scg_R = torch.einsum("bk,kij->bij", placement_probs, self.scg_rotations)   # (B, 3, 3)
        gyro_R = torch.einsum("bk,kij->bij", placement_probs, self.gyro_rotations)  # (B, 3, 3)

        scg = x[:, 0:3, :]    # (B, 3, T)
        gyro = x[:, 3:6, :]   # (B, 3, T)

        scg_corrected = torch.bmm(scg_R, scg)     # (B,3,3) @ (B,3,T) -> (B,3,T)
        gyro_corrected = torch.bmm(gyro_R, gyro)

        return torch.cat([scg_corrected, gyro_corrected], dim=1)

    def forward_hard_label(self, x: torch.Tensor, placement_idx: torch.Tensor) -> torch.Tensor:
        """Convenience path for inference when you have a hard predicted class index per sample, not a distribution."""
        probs = F.one_hot(placement_idx, num_classes=self.num_classes).float()
        return self.forward(x, probs)
