"""
src/utils/preprocessing.py

Signal-processing utilities shared by the data loader and the inference
pipeline:
  - resampling raw SCG streams to a fixed sampling rate
  - band-pass filtering (cardiac band)
  - fixed-length windowing
  - derivation of a 3-channel "gyro proxy" from the uncalibrated SCG stream

On the gyro proxy
------------------
The released MSCardio dataset currently ships calibrated + uncalibrated SCG
(accelerometer) signals; true gyroscope (GCG) channels are not yet public.
Per the project roadmap, we do NOT pretend to have real gyro data. Instead
we derive a 3-channel proxy from the uncalibrated SCG stream that captures
the slow drift/respiration/motion component the gyro channels would mostly
carry (low-frequency energy that calibration normally removes). This keeps
the 6-channel architecture exercised end-to-end today, and the proxy is
swapped for real GCG channels in `build_multimodal_manifest` the moment
that data is released -- search for "GYRO PROXY" to find the single place
this swap happens.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from src.config import SAMPLING_RATE_HZ, WINDOW_SAMPLES


def resample_signal(x: np.ndarray, orig_rate: float, target_rate: float = SAMPLING_RATE_HZ) -> np.ndarray:
    """Resample a (C, T) or (T,) signal from orig_rate to target_rate using polyphase filtering."""
    if orig_rate == target_rate:
        return x
    x = np.atleast_2d(x)
    duration = x.shape[-1] / orig_rate
    new_len = int(round(duration * target_rate))
    # resample_poly is more numerically stable than naive FFT resample for long signals
    gcd = np.gcd(int(round(orig_rate)), int(round(target_rate)))
    up = int(round(target_rate)) // gcd
    down = int(round(orig_rate)) // gcd
    out = sp_signal.resample_poly(x, up, down, axis=-1)
    if out.shape[-1] != new_len:
        # resample_poly's exact output length can differ by a sample or two; trim/pad
        if out.shape[-1] > new_len:
            out = out[..., :new_len]
        else:
            pad = new_len - out.shape[-1]
            out = np.pad(out, [(0, 0)] * (out.ndim - 1) + [(0, pad)], mode="edge")
    return out


def bandpass_cardiac(x: np.ndarray, fs: float = SAMPLING_RATE_HZ,
                      low_hz: float = 0.8, high_hz: float = 30.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass restricted to the cardiac vibration band."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 0.999)
    sos = sp_signal.butter(order, [low, high], btype="bandpass", output="sos")
    return sp_signal.sosfiltfilt(sos, x, axis=-1)


def derive_gyro_proxy(uncal_scg: np.ndarray, cal_scg: np.ndarray, fs: float = SAMPLING_RATE_HZ) -> np.ndarray:
    """
    GYRO PROXY: derive a 3-channel motion/respiration proxy from the
    difference between uncalibrated and calibrated SCG.

    Calibration in the MSCardio pipeline removes slow drift, gravity
    component changes, and gross motion artifact -- exactly the kind of
    signal a co-located gyroscope would mostly pick up. We isolate that
    removed component and low-pass it to ~2 Hz, which is in the range of
    respiration and postural motion (not cardiac micro-vibration), making
    it a reasonable physical stand-in for the gyro channel until real GCG
    data is released.

    Returns an array shaped like cal_scg (3, T).
    """
    diff = uncal_scg - cal_scg
    nyq = fs / 2.0
    sos = sp_signal.butter(3, min(2.0 / nyq, 0.999), btype="lowpass", output="sos")
    proxy = sp_signal.sosfiltfilt(sos, diff, axis=-1)
    return proxy


def normalize_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance normalize each channel independently."""
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True) + eps
    return (x - mu) / sigma


def make_fixed_windows(x: np.ndarray, window_samples: int = WINDOW_SAMPLES,
                        stride: int | None = None) -> np.ndarray:
    """
    Slice a (C, T) signal into overlapping fixed-length windows.
    Returns (num_windows, C, window_samples). Drops any trailing partial window.
    """
    if stride is None:
        stride = window_samples  # non-overlapping by default
    c, t = x.shape
    if t < window_samples:
        # pad with edge values if a recording is shorter than one window
        pad = window_samples - t
        x = np.pad(x, ((0, 0), (0, pad)), mode="edge")
        t = window_samples
    n_windows = 1 + (t - window_samples) // stride
    windows = np.stack([x[:, i * stride: i * stride + window_samples] for i in range(n_windows)])
    return windows


def assemble_six_channel(cal_scg: np.ndarray, uncal_scg: np.ndarray,
                          gyro: np.ndarray | None = None, fs: float = SAMPLING_RATE_HZ) -> np.ndarray:
    """
    Build the canonical (6, T) [SCG_x,y,z, Gyro_x,y,z] array used throughout
    the model. If `gyro` is None (current dataset release), falls back to
    the derived GYRO PROXY. Swap this for real GCG loading once released.
    """
    scg_clean = bandpass_cardiac(cal_scg, fs=fs)
    if gyro is None:
        gyro_channels = derive_gyro_proxy(uncal_scg, cal_scg, fs=fs)
    else:
        gyro_channels = gyro
    six_ch = np.concatenate([scg_clean, gyro_channels], axis=0)
    return six_ch
