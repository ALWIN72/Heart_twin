"""
src/heartprint/quality.py

Window-level signal quality assessment (SQI) and quality-aware window weighting.
Evaluates physiological cardiac signals (SCG + Gyro/Proxy) across multiple indices:
  - bSQI: Spectral energy ratio in cardiac band (0.8 - 30 Hz) vs noise/drift
  - pSQI: Periodic autocorrelation strength in physiological cardiac lag range (40 - 180 BPM)
  - kSQI: Kurtosis / fiducial complex sharpness
  - mSQI: Motion artifact, jerk, and amplitude sanity index

Generates quality scores in [0.0, 1.0] and continuous weights for numerical aggregation.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from src.config import SAMPLING_RATE_HZ, HeartPrintConfig, HEARTPRINT_CFG


class WindowQualityAssessor:
    """
    Assesses signal quality per window and derives non-linear contribution weights.
    """

    def __init__(self, cfg: HeartPrintConfig | None = None, fs: float = SAMPLING_RATE_HZ):
        self.cfg = cfg or HEARTPRINT_CFG
        self.fs = float(fs)

    def compute_window_sqi(self, window: np.ndarray) -> dict[str, float]:
        """
        Calculates SQI metrics for a single window array of shape (C, T) or (T,).
        Returns a dict with individual SQI metrics and the overall composite 'sqi' in [0, 1].
        """
        arr = np.asarray(window, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]

        # Evaluate primarily on the SCG accelerometer channels (first 3 channels or all if <3)
        scg_channels = arr[:min(3, arr.shape[0])]
        
        # 1. Amplitude & Flatline / Saturation check
        stds = np.std(scg_channels, axis=-1)
        means = np.mean(scg_channels, axis=-1)
        max_abs = np.max(np.abs(scg_channels), axis=-1)

        # Check for degenerate signals
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return {"sqi": 0.0, "bsqi": 0.0, "psqi": 0.0, "ksqi": 0.0, "msqi": 0.0, "usable": False}

        avg_std = float(np.mean(stds))
        if avg_std < 1e-4:  # Flatline or zeroed channel
            return {"sqi": 0.0, "bsqi": 0.0, "psqi": 0.0, "ksqi": 0.0, "msqi": 0.0, "usable": False}

        # 2. bSQI: Spectral energy ratio in cardiac band (0.8 - 30 Hz)
        # Compute Welch power spectral density on SCG norm
        scg_mag = np.linalg.norm(scg_channels, axis=0) if scg_channels.shape[0] > 1 else scg_channels[0]
        scg_mag_centered = scg_mag - np.mean(scg_mag)
        
        nperseg = min(len(scg_mag_centered), int(self.fs * 4))
        if nperseg < int(self.fs):
            nperseg = len(scg_mag_centered)
            
        freqs, psd = sp_signal.welch(scg_mag_centered, fs=self.fs, nperseg=nperseg)
        total_power = np.sum(psd) + 1e-12
        cardiac_band_mask = (freqs >= 0.8) & (freqs <= 30.0)
        cardiac_power = np.sum(psd[cardiac_band_mask])
        bsqi = float(np.clip(cardiac_power / total_power, 0.0, 1.0))

        # 3. pSQI: Periodicity index (Autocorrelation in 40-180 BPM lag range)
        # 40 BPM = 1.5s lag, 180 BPM = 0.33s lag
        min_lag = max(1, int(0.33 * self.fs))
        max_lag = min(len(scg_mag_centered) - 1, int(1.5 * self.fs))
        
        if max_lag > min_lag:
            # Lowpass filter the envelope to capture heartbeat periodicity
            nyq = self.fs / 2.0
            sos = sp_signal.butter(3, min(5.0 / nyq, 0.99), btype="lowpass", output="sos")
            env = sp_signal.sosfiltfilt(sos, scg_mag_centered ** 2)
            env_centered = env - np.mean(env)
            
            ac0 = np.sum(env_centered ** 2) + 1e-12
            autocorr = np.correlate(env_centered, env_centered, mode="full")
            mid = len(autocorr) // 2
            lags_corr = autocorr[mid + min_lag : mid + max_lag + 1] / ac0
            max_ac = float(np.max(lags_corr)) if len(lags_corr) > 0 else 0.0
            psqi = float(np.clip(max_ac, 0.0, 1.0))
        else:
            psqi = 0.0

        # 4. kSQI: Kurtosis of the signal (cardiac fiducial pulses yield kurtosis > 3.0)
        m2 = np.mean(scg_mag_centered ** 2) + 1e-12
        m4 = np.mean(scg_mag_centered ** 4)
        kurt = m4 / (m2 ** 2)  # Normal Gaussian kurtosis = 3.0
        # Scale kurtosis: kurt < 2.0 (flat noise) -> 0.2, kurt ~ 3.5-8.0 -> 0.8-1.0
        ksqi = float(np.clip((kurt - 1.5) / 5.0, 0.0, 1.0))

        # 5. mSQI: Motion jerk / spike detection
        # Sudden massive spikes (jerk) indicate phone shifting or tapping
        diffs = np.diff(scg_channels, axis=-1)
        jerk_metric = np.mean(np.abs(diffs)) / (avg_std + 1e-8)
        # Moderate jerk is normal, huge jerk (> 1.5 std/sample) indicates motion artifact
        if jerk_metric > 1.2:
            msqi = float(np.clip(1.0 - (jerk_metric - 1.2) * 1.5, 0.0, 1.0))
        else:
            msqi = 1.0

        # Composite SQI (weighted average of physical quality indices)
        composite = 0.35 * bsqi + 0.35 * psqi + 0.15 * ksqi + 0.15 * msqi
        composite = float(np.clip(composite, 0.0, 1.0))

        is_usable = bool(composite >= self.cfg.min_window_quality)

        return {
            "sqi": round(composite, 4),
            "bsqi": round(bsqi, 4),
            "psqi": round(psqi, 4),
            "ksqi": round(ksqi, 4),
            "msqi": round(msqi, 4),
            "usable": is_usable,
        }

    def compute_quality_weights(self, qualities: list[float] | np.ndarray) -> np.ndarray:
        """
        Converts window SQI scores into non-linear contribution weights.
        Unusable windows (< min_window_quality) receive weight = 0.0.
        Usable windows receive weights smoothly scaled from (0.0, 1.0].
        """
        q_arr = np.asarray(qualities, dtype=np.float64)
        q_min = self.cfg.min_window_quality
        weights = np.zeros_like(q_arr)

        mask = q_arr >= q_min
        if np.any(mask):
            # Smooth power curve weighting
            scaled = (q_arr[mask] - q_min) / max(1e-6, (1.0 - q_min))
            weights[mask] = np.clip(scaled ** 1.3, 0.05, 1.0)

        return weights

    def evaluate_recording_viability(self, window_sqis: list[dict]) -> dict:
        """
        Determines if a full recording meets the threshold of usable windows
        to participate in HeartPrint enrollment and Heart Twin updating.
        """
        total = len(window_sqis)
        if total == 0:
            return {
                "viable": False,
                "usable_count": 0,
                "total_count": 0,
                "usable_ratio": 0.0,
                "median_quality": 0.0,
                "reason": "No windows found in recording.",
            }

        sqi_values = [w["sqi"] for w in window_sqis]
        usable_count = sum(1 for w in window_sqis if w["usable"])
        usable_ratio = usable_count / total
        med_quality = float(np.median(sqi_values))

        viable = (
            usable_count >= self.cfg.min_valid_windows
            and usable_ratio >= self.cfg.min_usable_window_ratio
        )

        reason = "Viable recording with sufficient reliable cardiac signal."
        if not viable:
            if usable_count < self.cfg.min_valid_windows:
                reason = (
                    f"Insufficient reliable signal (only {usable_count}/{total} usable windows; "
                    f"minimum required is {self.cfg.min_valid_windows}). "
                    f"Please keep the smartphone steady on your chest and record again."
                )
            else:
                reason = (
                    f"Usable window ratio ({usable_ratio * 100:.1f}%) is below minimum "
                    f"threshold ({self.cfg.min_usable_window_ratio * 100:.0f}%). "
                    f"Please ensure steady contact and record again."
                )

        return {
            "viable": viable,
            "usable_count": usable_count,
            "total_count": total,
            "usable_ratio": round(usable_ratio, 4),
            "median_quality": round(med_quality, 4),
            "reason": reason,
        }
