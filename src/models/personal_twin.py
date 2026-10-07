"""
src/models/personal_twin.py

Phase 4: Personalized Digital Heart Twin (the core innovation).

A small per-user VAE head sits on top of the frozen global foundation
encoder. It is trained ONLY on that user's own baseline recordings, so it
learns "what normal looks like" for that specific person. After the
baseline distribution is established, new recordings are scored by how
far their latent encoding deviates (KL divergence) from the user's stored
baseline distribution -- a personalized anomaly score, not a population
comparison.

Differences from the original sketch
-------------------------------------
- `forward(..., store_baseline=True)` no longer mutates persistent buffers
  as a side effect of a single call. Storing a baseline is each user's most
  important piece of state (their entire "normal" reference), so it's
  exposed as an explicit `set_baseline(...)` method that aggregates over a
  *batch* of baseline windows, not a quietly-overwritten running value
  recomputed (and silently lost) on every call.
- The KL-divergence anomaly score is now computed per-sample, not summed
  in a way that hides batch structure.
- The decoder actually reconstructs the *global encoder's* 256-d feature
  (matching the spec's stated architecture), and the reconstruction loss
  used during baseline fine-tuning is defined consistently with that.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import ModelConfig, TwinConfig
from src.models.encoder import SCGEncoder


class PersonalHeartTwin(nn.Module):
    def __init__(self, global_encoder: SCGEncoder, user_id: str,
                 model_cfg: ModelConfig | None = None, twin_cfg: TwinConfig | None = None):
        super().__init__()
        self.user_id = user_id
        self.model_cfg = model_cfg = model_cfg or global_encoder.cfg
        self.twin_cfg = twin_cfg = twin_cfg or TwinConfig()

        # 1. Freeze the global foundation (trained on all users).
        self.global_encoder = global_encoder
        for param in self.global_encoder.parameters():
            param.requires_grad = False

        # 2. Personal VAE head (trained ONLY on this user's baseline recordings).
        self.personal_encoder = nn.Sequential(
            nn.Linear(model_cfg.embed_dim, twin_cfg.personal_hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(twin_cfg.personal_hidden_dim, twin_cfg.latent_dim)
        self.fc_logvar = nn.Linear(twin_cfg.personal_hidden_dim, twin_cfg.latent_dim)

        # 3. Decoder to reconstruct the user's global feature vector from the latent.
        self.decoder = nn.Sequential(
            nn.Linear(twin_cfg.latent_dim, twin_cfg.personal_hidden_dim), nn.ReLU(),
            nn.Linear(twin_cfg.personal_hidden_dim, model_cfg.embed_dim),
        )

        # Baseline distribution, established once via set_baseline().
        self.register_buffer("baseline_mu", torch.zeros(1, twin_cfg.latent_dim))
        self.register_buffer("baseline_logvar", torch.zeros(1, twin_cfg.latent_dim))
        self.register_buffer("has_baseline", torch.tensor(False))
        self.register_buffer("threshold", torch.tensor(twin_cfg.anomaly_threshold, dtype=torch.float32))


    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, in_channels, T) -> global_feat, mu, logvar."""
        with torch.no_grad():
            global_feat = self.global_encoder.encode_pooled(x)  # (B, embed_dim)
        personal = self.personal_encoder(global_feat)
        mu = self.fc_mu(personal)
        logvar = self.fc_logvar(personal)
        return global_feat, mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        """
        Standard VAE forward pass for training: returns reconstruction,
        global_feat target, mu, logvar so the caller can compute
        recon + KL(q(z|x) || N(0,1)) loss.
        """
        global_feat, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {"recon": recon, "target": global_feat, "mu": mu, "logvar": logvar, "z": z}

    @torch.no_grad()
    def set_baseline(self, baseline_windows: torch.Tensor):
        """
        baseline_windows: (B, in_channels, T) -- a batch of this user's
        known-healthy baseline recordings. Aggregates their (mu, logvar)
        into a single stored "normal" distribution for this user.
        Correctly accounts for both within-sample and between-sample variance.
        """
        self.eval()
        _, mu, logvar = self.encode(baseline_windows)
        self.baseline_mu = mu.mean(dim=0, keepdim=True).clone()
        
        # Total pooled variance of a Gaussian mixture: within-sample + between-sample
        within_var = torch.exp(logvar).mean(dim=0, keepdim=True)
        between_var = ((mu - self.baseline_mu) ** 2).mean(dim=0, keepdim=True)
        var = within_var + between_var
        self.baseline_logvar = torch.log(var.clamp(min=1e-8))
        self.has_baseline.fill_(True)

        # Calibrate default threshold on the baseline itself (e.g. 98th percentile)
        scores = self.anomaly_score(baseline_windows).cpu().numpy()
        if len(scores) > 0:
            pct_val = float(np.percentile(scores, self.twin_cfg.calibration_percentile))
            sigma_val = float(np.mean(scores) + self.twin_cfg.calibration_sigma_k * np.std(scores))
            self.threshold.fill_(max(pct_val, sigma_val, 1e-4))

    @torch.no_grad()
    def calibrate_threshold(self, validation_windows: torch.Tensor, percentile: float | None = None,
                            sigma_k: float | None = None) -> float:
        """
        Calibrate user-specific anomaly threshold on held-out healthy enrollment/validation data.
        Guarantees that healthy baseline achieves target specificity (~2% false alarm).
        """
        pct = percentile if percentile is not None else self.twin_cfg.calibration_percentile
        k = sigma_k if sigma_k is not None else self.twin_cfg.calibration_sigma_k
        scores = self.anomaly_score(validation_windows).cpu().numpy()
        if len(scores) == 0:
            return float(self.threshold.item())
        pct_val = float(np.percentile(scores, pct))
        sigma_val = float(np.mean(scores) + k * np.std(scores))
        chosen = max(pct_val, sigma_val, 1e-4)
        self.threshold.fill_(chosen)
        return chosen

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_channels, T). Returns (B,) per-sample KL divergence of
        this window's latent distribution from the stored baseline.
        Higher = more anomalous relative to this user's own history.
        Raises if no baseline has been set yet.
        """
        if not bool(self.has_baseline.item()):
            raise RuntimeError(
                f"PersonalHeartTwin for user {self.user_id} has no baseline set. "
                f"Call set_baseline(...) with this user's healthy recordings first."
            )
        self.eval()
        _, mu, logvar = self.encode(x)

        kl_div = 0.5 * torch.sum(
            torch.exp(logvar - self.baseline_logvar)
            + (self.baseline_mu - mu) ** 2 / torch.exp(self.baseline_logvar)
            - 1
            - (logvar - self.baseline_logvar),
            dim=-1,
        )
        return kl_div  # (B,)

