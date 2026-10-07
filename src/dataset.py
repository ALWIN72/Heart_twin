"""
src/dataset.py

PyTorch Dataset classes that turn manifest rows into windowed, 6-channel
(SCG + gyro-proxy) tensors ready for the foundation model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import N_CHANNELS, SAMPLING_RATE_HZ, WINDOW_SAMPLES
from src.utils.preprocessing import (
    assemble_six_channel,
    make_fixed_windows,
    normalize_per_channel,
    resample_signal,
)


def _load_scg_csv(path: str, expected_rate_col: str = "sampling_rate") -> tuple[np.ndarray, float]:
    """
    Load a scg.csv / Uncalibrated_scg.csv file.

    Handles two common layouts:
      1. Columns named x,y,z (+ optional timestamp) -> shape (3, T)
      2. A single value column (already flattened) -> shape (1, T), will be
         tiled to (3, T) with a loud warning, since the model expects 3 axes.

    Sampling rate is inferred from a 'timestamp' column if present,
    otherwise defaults to SAMPLING_RATE_HZ (already-resampled assumption).
    """
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    axis_cols = [cols_lower[a] for a in ("x", "y", "z") if a in cols_lower]
    if len(axis_cols) == 3:
        arr = df[axis_cols].to_numpy().T  # (3, T)
    else:
        # Fall back: use all numeric, non-timestamp columns
        numeric_cols = [c for c in df.columns
                         if c.lower() not in ("timestamp", "time", "t", "seconds_elapsed")
                         and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 3:
            axis_cols = numeric_cols[:3]
            arr = df[axis_cols].to_numpy().T
        elif len(numeric_cols) >= 1:
            arr = df[numeric_cols[0]].to_numpy()[None, :]
            arr = np.repeat(arr, 3, axis=0)
        else:
            raise ValueError(f"Could not find usable signal columns in {path}; got columns {list(df.columns)}")

    fs = SAMPLING_RATE_HZ
    # Prefer explicit seconds column when present
    ts_col = (cols_lower.get("seconds_elapsed") or cols_lower.get("timestamp") or cols_lower.get("time"))
    if ts_col is not None and len(df) > 1:
        t = df[ts_col].to_numpy().astype(float)
        dt = float(np.median(np.diff(t)))
        if dt > 0:
            for scale in (1.0, 1e3, 1e9):   # seconds, milliseconds, nanoseconds
                cand = scale / dt
                if 1.0 < cand < 5000.0:
                    fs = cand
                    break

    return arr.astype(np.float32), float(fs)


class SCGWindowDataset(Dataset):
    """
    Produces fixed-length (N_CHANNELS, WINDOW_SAMPLES) windows from every
    recording in a manifest DataFrame. Each window retains full provenance:
    (subject_id, recording_id, session_id, window_id, timing, signal quality).
    """

    def __init__(self, manifest_df: pd.DataFrame, window_samples: int = WINDOW_SAMPLES,
                 stride: int | None = None, normalize: bool = True, cache_in_memory: bool = True,
                 min_quality: float = 0.0, assess_quality: bool = True):
        self.manifest = manifest_df.reset_index(drop=True)
        self.window_samples = window_samples
        self.stride = stride if stride is not None else window_samples
        self.normalize = normalize
        self.cache_in_memory = cache_in_memory
        self.min_quality = float(min_quality)
        self.assess_quality = assess_quality
        if self.assess_quality or self.min_quality > 0.0:
            from src.heartprint.quality import WindowQualityAssessor
            self.quality_assessor = WindowQualityAssessor(fs=SAMPLING_RATE_HZ)
        else:
            self.quality_assessor = None
        self._cache: dict[int, np.ndarray] = {}
        self._sqi_cache: dict[int, list[dict]] = {}
        self._index = self._build_index()

    def _build_index(self):
        """Pre-scan recordings to know how many windows each contributes, respecting stride."""
        index = []
        for row_i, row in self.manifest.iterrows():
            try:
                cal, fs_cal = _load_scg_csv(row["cal_scg"])
            except Exception as e:
                print(f"[SCGWindowDataset] Skipping unreadable recording "
                      f"{row.get('subject_id')}/{row.get('recording_id')}: {e}")
                continue
            n_samples_resampled = int(round(cal.shape[-1] * SAMPLING_RATE_HZ / fs_cal))
            if n_samples_resampled < self.window_samples:
                n_windows = 1
            else:
                n_windows = 1 + (n_samples_resampled - self.window_samples) // self.stride

            for w in range(n_windows):
                index.append((row_i, w))

        # Filter by quality if requested
        if self.min_quality > 0.0:
            filtered_index = []
            for row_i, w in index:
                sqis = self._get_or_compute_sqis(row_i)
                if w < len(sqis) and sqis[w]["sqi"] >= self.min_quality:
                    filtered_index.append((row_i, w))
            return filtered_index

        return index

    def __len__(self):
        return len(self._index)

    def _get_or_compute_sqis(self, row_i: int) -> list[dict]:
        if row_i in self._sqi_cache:
            return self._sqi_cache[row_i]
        windows = self._load_windows(row_i)
        sqis = []
        for w in range(windows.shape[0]):
            info = self.quality_assessor.compute_window_sqi(windows[w])
            weight = float(self.quality_assessor.compute_quality_weights([info["sqi"]])[0])
            info["quality_weight"] = weight
            sqis.append(info)
        self._sqi_cache[row_i] = sqis
        return sqis

    def _load_windows(self, row_i: int) -> np.ndarray:
        if self.cache_in_memory and row_i in self._cache:
            return self._cache[row_i]

        row = self.manifest.iloc[row_i]
        cal_raw, fs_cal = _load_scg_csv(row["cal_scg"])
        cal = resample_signal(cal_raw, fs_cal, SAMPLING_RATE_HZ)

        if row.get("uncal_scg") and isinstance(row["uncal_scg"], str) and Path(row["uncal_scg"]).exists():
            uncal_raw, fs_uncal = _load_scg_csv(row["uncal_scg"])
            uncal = resample_signal(uncal_raw, fs_uncal, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], uncal.shape[-1])
            cal, uncal = cal[:, :min_len], uncal[:, :min_len]
        else:
            uncal = cal.copy()

        gyro_real = None
        if row.get("gyro") and isinstance(row["gyro"], str) and Path(row["gyro"]).exists():
            gyro_raw, fs_gyro = _load_scg_csv(row["gyro"])
            gyro_real = resample_signal(gyro_raw, fs_gyro, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], gyro_real.shape[-1])
            gyro_real = gyro_real[:, :min_len]
            cal = cal[:, :min_len]
            uncal = uncal[:, :min_len]

        six_ch = assemble_six_channel(cal, uncal, gyro=gyro_real, fs=SAMPLING_RATE_HZ)
        if self.normalize:
            six_ch = normalize_per_channel(six_ch)

        windows = make_fixed_windows(six_ch, window_samples=self.window_samples, stride=self.stride)

        if self.cache_in_memory:
            self._cache[row_i] = windows
        return windows

    def __getitem__(self, idx: int):
        row_i, w = self._index[idx]
        windows = self._load_windows(row_i)
        w = min(w, windows.shape[0] - 1)
        x = windows[w]  # (N_CHANNELS, window_samples)

        row = self.manifest.iloc[row_i]
        sqis = self._get_or_compute_sqis(row_i) if self.assess_quality else []
        sqi_info = sqis[w] if w < len(sqis) else {"sqi": 1.0, "usable": True, "quality_weight": 1.0}

        sub_id = str(row["subject_id"])
        rec_id = str(row["recording_id"])
        sess_id = str(row.get("session_id", "session_01"))
        start_s = round(w * self.stride / SAMPLING_RATE_HZ, 3)
        end_s = round((w * self.stride + self.window_samples) / SAMPLING_RATE_HZ, 3)

        meta = {
            "subject_id": sub_id,
            "recording_id": rec_id,
            "session_id": sess_id,
            "window_id": f"{sub_id}_{rec_id}_w{w:03d}",
            "window_idx": w,
            "start_time_s": start_s,
            "end_time_s": end_s,
            "recording_time": str(row.get("recording_time", "")),
            "recording_timezone": str(row.get("recording_timezone", "")),
            "sqi": sqi_info.get("sqi", 1.0),
            "bsqi": sqi_info.get("bsqi", 1.0),
            "psqi": sqi_info.get("psqi", 1.0),
            "ksqi": sqi_info.get("ksqi", 1.0),
            "msqi": sqi_info.get("msqi", 1.0),
            "usable": sqi_info.get("usable", True),
            "quality_weight": sqi_info.get("quality_weight", 1.0),
            "platform": str(row.get("platform", "unknown")),
            "device": str(row.get("device", "unknown")),
            "placement": str(row.get("placement", "unknown")),
        }
        return torch.from_numpy(x).float(), meta


def collate_scg(batch):
    """Custom collate that keeps metadata as a list of dicts instead of trying to tensorize it."""
    xs = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    return xs, metas

