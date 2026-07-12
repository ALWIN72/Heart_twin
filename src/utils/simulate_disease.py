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


def simulate_heart_deterioration(healthy_signal: np.ndarray, days: int = 30,
                                  onset_day: int = 20, seed: int | None = None) -> Iterator[np.ndarray]:
    """
    healthy_signal: (6, T) a single known-healthy window for one subject.
    Yields `days` synthetic windows, day 0 == the original healthy signal,
    gradually drifting after `onset_day` to simulate symptom onset.

    Drift components (all ramp in only after onset_day, scaled by how far
    past onset we are):
      - amplitude decay (contractility loss)
      - timing jitter (irregular rhythm) via local time-warping
      - added broadband noise (artifact / murmur proxy)
    """
    rng = np.random.default_rng(seed)
    c, t = healthy_signal.shape

    for day in range(days):
        if day < onset_day:
            # Pre-onset: small natural day-to-day variation only, no systematic drift.
            noise = rng.normal(0, 0.02, size=healthy_signal.shape)
            yield (healthy_signal + noise).astype(np.float32)
            continue

        severity = (day - onset_day + 1) / max(1, (days - onset_day))  # 0 -> 1 over the post-onset period
        sig = healthy_signal.copy()

        # 1. Amplitude decay
        amplitude_factor = 1.0 - 0.4 * severity
        sig = sig * amplitude_factor

        # 2. Timing jitter via local resampling of a random subsegment
        if severity > 0.1:
            jitter_strength = 0.15 * severity
            idx = np.arange(t)
            warp = idx + (jitter_strength * t) * np.sin(2 * np.pi * idx / t * rng.uniform(2, 5))
            warp = np.clip(warp, 0, t - 1)
            sig = sig[:, warp.astype(int)]

        # 3. Added broadband noise (artifact/murmur proxy)
        noise_std = 0.03 + 0.25 * severity
        sig = sig + rng.normal(0, noise_std, size=sig.shape)

        yield sig.astype(np.float32)
