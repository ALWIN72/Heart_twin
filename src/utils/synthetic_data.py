"""
src/utils/synthetic_data.py

DEV/TEST FIXTURE ONLY -- not used by any production training or inference
path. Generates a small, plausible-looking SCG dataset on disk in the exact
directory layout `build_multimodal_manifest` expects, so the rest of the
pipeline (manifest -> dataset -> models -> training -> eval -> dashboard)
can be exercised and unit-tested before the real MSCardio data is mounted.

Replace/remove this once data/raw/MSCardio contains the real dataset --
`build_multimodal_manifest` works identically on real or synthetic data
because it only depends on the directory layout, not the content.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_RAW, SAMPLING_RATE_HZ


def _synthetic_heartbeat_train(n_samples: int, fs: float, hr_bpm: float = 65,
                                amplitude: float = 1.0, noise_std: float = 0.05,
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a 3-axis pseudo-SCG signal: quasi-periodic AO/MC complexes + noise."""
    rng = rng or np.random.default_rng()
    t = np.arange(n_samples) / fs
    beat_period = 60.0 / hr_bpm

    sig = np.zeros((3, n_samples))
    beat_times = np.arange(0, t[-1], beat_period)
    for bt in beat_times:
        # jitter each beat slightly for realism
        bt_j = bt + rng.normal(0, 0.01)
        # AO complex: damped oscillation ~10-25 Hz shortly after beat onset
        for axis in range(3):
            freq = rng.uniform(10, 20)
            decay = rng.uniform(15, 25)
            phase = rng.uniform(0, np.pi)
            envelope = amplitude * (0.7 + 0.3 * rng.random()) * np.exp(-decay * np.clip(t - bt_j, 0, None)) \
                * np.sin(2 * np.pi * freq * (t - bt_j) + phase)
            envelope[t < bt_j] = 0
            sig[axis] += envelope

    sig += rng.normal(0, noise_std, size=sig.shape)
    # slow respiration drift (~0.25 Hz)
    resp = 0.15 * amplitude * np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, np.pi))
    sig += resp[None, :]
    return sig.astype(np.float32)


def generate_synthetic_dataset(
    out_root: Path | str = DATA_RAW,
    n_subjects: int = 6,
    recordings_per_subject: int = 8,
    duration_seconds: int = 30,
    fs: float = SAMPLING_RATE_HZ,
    seed: int = 0,
) -> Path:
    """
    Writes a synthetic MSCardio-shaped dataset to `out_root/MSCardio/...`.
    Returns the root path written.
    """
    rng = np.random.default_rng(seed)
    root = Path(out_root) / "MSCardio"
    root.mkdir(parents=True, exist_ok=True)

    platforms = ["iOS", "Android"]
    devices = {"iOS": ["iPhone 13", "iPhone 14"], "Android": ["Pixel 7", "Galaxy S22"]}

    for s in range(1, n_subjects + 1):
        subj_id = f"{s:03d}"
        subj_dir = root / f"Subject_{subj_id}"
        subj_dir.mkdir(exist_ok=True)

        platform = platforms[s % 2]
        device = devices[platform][s % 2]
        base_hr = rng.uniform(55, 85)

        meta = {
            "smartphone_model": device,
            "platform": platform,
            "gender": "F" if s % 2 == 0 else "M",
            "age": int(rng.integers(22, 70)),
            "height": round(float(rng.uniform(155, 190)), 1),
            "weight": round(float(rng.uniform(55, 95)), 1),
        }
        with open(subj_dir / "general_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        n_samples = int(duration_seconds * fs)
        for r in range(1, recordings_per_subject + 1):
            rec_id = f"{r:02d}"
            rec_dir = subj_dir / f"Recording_{rec_id}"
            rec_dir.mkdir(exist_ok=True)

            # Slight per-recording HR drift to simulate day-to-day variation
            hr = base_hr + rng.normal(0, 2)
            clean = _synthetic_heartbeat_train(n_samples, fs, hr_bpm=hr, amplitude=1.0,
                                                noise_std=0.04, rng=rng)

            # "Uncalibrated" = clean + extra low-freq drift/motion that
            # calibration would normally strip out (this is the component
            # derive_gyro_proxy() recovers).
            t = np.arange(n_samples) / fs
            drift = 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.15) * t + rng.uniform(0, np.pi))
            motion_burst = np.zeros(n_samples)
            if rng.random() < 0.3:
                start = rng.integers(0, max(1, n_samples - int(fs * 2)))
                motion_burst[start:start + int(fs * 2)] = rng.normal(0, 0.3, size=int(fs * 2))
            uncal = clean + (drift + motion_burst)[None, :]

            cal_df = pd.DataFrame(clean.T, columns=["x", "y", "z"])
            cal_df.insert(0, "timestamp", t)
            cal_df.to_csv(rec_dir / "scg.csv", index=False)

            uncal_df = pd.DataFrame(uncal.T, columns=["x", "y", "z"])
            uncal_df.insert(0, "timestamp", t)
            uncal_df.to_csv(rec_dir / "Uncalibrated_scg.csv", index=False)

    return root


if __name__ == "__main__":
    path = generate_synthetic_dataset()
    print(f"Synthetic dev dataset written to {path}")
