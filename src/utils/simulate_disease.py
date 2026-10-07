"""
src/utils/simulate_disease.py

Synthetic cardiac deterioration simulator for early-warning evaluation
(Phase 5). We don't have longitudinal data showing real users transition
from healthy to symptomatic, so to evaluate whether the system *could*
catch deterioration early, we simulate a plausible day-by-day drift away
from a known-healthy baseline window: progressively reduced amplitude
(weaker contractility), increasing irregularity (timing jitter), and
added high-frequency noise (murmur-like artifact). This is a stand-in
stress test, not a validated clinical model -- it exercises the detection
pipeline's sensitivity, it does not claim to model any specific
pathology.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np


from src.utils.preprocessing import normalize_per_channel


def simulate_heart_deterioration(healthy_signal: np.ndarray, days: int = 30,
                                 onset_day: int = 20, seed: int | None = None) -> Iterator[np.ndarray]:
    """
    healthy_signal: (6, T) a single known-healthy window for one subject.
    Yields `days` synthetic windows, day 0 == the original healthy signal,
    gradually drifting after `onset_day` to simulate symptom onset.

    Drift components (ramp in progressively after onset_day):
      - systolic peak blunting (contractility loss via non-linear wave compression)
      - abnormal diastolic resonance / murmurs (elevated pathological harmonic frequencies)
      - timing jitter (irregular rhythm via local time-warping)
      - broadband acoustic noise (loss of harmonic coherence)
      - canonical per-channel normalization
    """
    rng = np.random.default_rng(seed)
    c, t = healthy_signal.shape

    for day in range(days):
        if day < onset_day:
            # Pre-onset: natural physiological day-to-day variation only, no systematic drift.
            noise = rng.normal(0, 0.03, size=healthy_signal.shape)
            varied = healthy_signal + noise
            yield normalize_per_channel(varied).astype(np.float32)
            continue

        severity = (day - onset_day + 1) / max(1, (days - onset_day))  # 0 -> 1 over post-onset
        sig = healthy_signal.copy()

        # 1. Systolic blunting: non-linear compression blunts peak aortic ejection contractility
        sig = np.tanh(sig * (1.0 - 0.45 * severity))

        # 2. Timing jitter: local time-warping representing conduction delay / arrhythmia
        if severity > 0.05:
            jitter_strength = 0.12 * severity
            idx = np.arange(t)
            warp = idx + (jitter_strength * t) * np.sin(2 * np.pi * idx / t * rng.uniform(2, 5))
            warp = np.clip(warp, 0, t - 1)
            sig = sig[:, warp.astype(int)]

        # 3. Pathological resonance / murmurs in diastolic window
        resonance = 0.25 * severity * np.sin(np.linspace(0, 32 * np.pi, t))
        sig = sig + resonance[np.newaxis, :]

        # 4. Added acoustic noise / turbulence
        noise_std = 0.03 + 0.12 * severity
        sig = sig + rng.normal(0, noise_std, size=sig.shape)

        # 5. Canonical per-channel normalization matching pipeline inference
        yield normalize_per_channel(sig).astype(np.float32)

