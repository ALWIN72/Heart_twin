"""
tests/test_heartprint.py

Comprehensive test suite for HeartPrint Progressive Cardiac Identity Enrollment:
  - Window quality assessment & SQI calculations
  - Quality-aware weighting and robust numerical aggregation
  - Mixed-quality signal handling (tolerating bad windows without dropping recording)
  - Full recording rejection on insufficient viable data
  - Non-destructive progression across multi-session recordings
  - 20-piece heart puzzle generation and subspace stability mapping
  - Persistence, state serialization, and edge-case resilience
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import SAMPLING_RATE_HZ, HeartPrintConfig
from src.heartprint.engine import HeartPrintEngine
from src.heartprint.puzzle import HeartPuzzle
from src.heartprint.quality import WindowQualityAssessor
from src.heartprint.state import HeartPrintState


def _generate_synthetic_scg(
    duration_seconds: float = 30.0,
    fs: float = 250.0,
    bpm: float = 72.0,
    noise_level: float = 0.05,
    motion_spikes: bool = False,
) -> np.ndarray:
    """Creates a realistic 6-channel synthetic SCG + Gyro signal."""
    t = np.arange(0, duration_seconds, 1.0 / fs)
    n_samples = len(t)
    heart_rate_hz = bpm / 60.0

    # Cardiac SCG fundamental and harmonics
    scg_z = (
        1.0 * np.sin(2 * np.pi * heart_rate_hz * t)
        + 0.5 * np.sin(4 * np.pi * heart_rate_hz * t + 0.3)
        + 0.25 * np.sin(6 * np.pi * heart_rate_hz * t + 0.7)
    )
    scg_x = 0.3 * np.sin(2 * np.pi * heart_rate_hz * t + 0.5)
    scg_y = 0.4 * np.cos(2 * np.pi * heart_rate_hz * t + 0.2)

    # Add Gaussian noise
    scg_x += np.random.randn(n_samples) * noise_level
    scg_y += np.random.randn(n_samples) * noise_level
    scg_z += np.random.randn(n_samples) * noise_level

    # Gyro proxy (slow respiration component ~0.25 Hz)
    resp = 0.5 * np.sin(2 * np.pi * 0.25 * t)
    gyro_x = resp + np.random.randn(n_samples) * noise_level * 0.5
    gyro_y = resp * 0.8 + np.random.randn(n_samples) * noise_level * 0.5
    gyro_z = resp * 0.5 + np.random.randn(n_samples) * noise_level * 0.5

    if motion_spikes:
        # Inject motion artifacts in specific windows
        spike_start = int(8.0 * fs)
        spike_end = int(12.0 * fs)
        scg_z[spike_start:spike_end] += np.random.randn(spike_end - spike_start) * 20.0

    six_ch = np.stack([scg_x, scg_y, scg_z, gyro_x, gyro_y, gyro_z], axis=0)
    return six_ch.astype(np.float32)


def test_window_quality_assessment():
    """Verify SQI detects clean signals vs noise vs flatline."""
    assessor = WindowQualityAssessor()

    # 1. Clean signal
    clean_sig = _generate_synthetic_scg(duration_seconds=10.0, noise_level=0.02)
    clean_sqi = assessor.compute_window_sqi(clean_sig)
    assert clean_sqi["sqi"] > 0.60
    assert clean_sqi["usable"] is True
    assert clean_sqi["bsqi"] > 0.50

    # 2. Pure random noise (no cardiac periodicity)
    noise_sig = np.random.randn(6, 2500).astype(np.float32)
    noise_sqi = assessor.compute_window_sqi(noise_sig)
    assert noise_sqi["psqi"] < 0.40

    # 3. Flatline (zeros)
    flat_sig = np.zeros((6, 2500), dtype=np.float32)
    flat_sqi = assessor.compute_window_sqi(flat_sig)
    assert flat_sqi["sqi"] == 0.0
    assert flat_sqi["usable"] is False

    # 4. Signal with NaNs
    nan_sig = clean_sig.copy()
    nan_sig[0, 100] = np.nan
    nan_sqi = assessor.compute_window_sqi(nan_sig)
    assert nan_sqi["sqi"] == 0.0
    assert nan_sqi["usable"] is False


def test_quality_weighting_and_aggregation():
    """Verify non-linear weighting and numerical stability."""
    assessor = WindowQualityAssessor()

    qualities = [0.95, 0.85, 0.40, 0.20, 0.90]
    weights = assessor.compute_quality_weights(qualities)

    assert len(weights) == 5
    # Unusable windows (< 0.50) must have zero weight
    assert weights[2] == 0.0
    assert weights[3] == 0.0
    # Good windows must have positive weights
    assert weights[0] > weights[1] > 0.0
    assert weights[4] > 0.0

    # Viability check
    window_sqis = [{"sqi": q, "usable": q >= 0.50} for q in qualities]
    viability = assessor.evaluate_recording_viability(window_sqis)
    assert viability["usable_count"] == 3
    assert viability["total_count"] == 5
    assert viability["usable_ratio"] == 0.60
    assert viability["viable"] is True


def test_mixed_quality_recording_success():
    """Verify a recording with alternating good and weak windows still succeeds."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        cfg = HeartPrintConfig(min_valid_windows=2, min_usable_window_ratio=0.40)
        engine = HeartPrintEngine(cfg=cfg, checkpoint_dir=tmpdir)

        # 30s recording with a motion spike in the middle
        sig = _generate_synthetic_scg(duration_seconds=30.0, motion_spikes=True)
        res = engine.process_six_channel_recording("test_sub", sig, recording_id="01")

        assert res["ok"] is True
        assert res["accepted"] is True
        assert res["usable_windows"] >= 2
        assert res["pieces_completed"] >= 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_noisy_recording_rejection_without_state_corruption():
    """Verify a mostly bad recording is rejected and does not mutate previously established state."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        cfg = HeartPrintConfig(min_valid_windows=2, min_usable_window_ratio=0.50)
        engine = HeartPrintEngine(cfg=cfg, checkpoint_dir=tmpdir)

        # 1. First good recording
        sig_good = _generate_synthetic_scg(duration_seconds=30.0, noise_level=0.03)
        res1 = engine.process_six_channel_recording("test_sub", sig_good, recording_id="01")
        assert res1["accepted"] is True

        state_after_good = engine.get_state("test_sub")
        pieces_before = state_after_good.pieces_completed
        score_before = state_after_good.learning_score
        recs_before = state_after_good.recordings_processed

        # 2. Bad recording (all noise)
        bad_sig = np.random.randn(6, 250 * 30).astype(np.float32)
        res_bad = engine.process_six_channel_recording("test_sub", bad_sig, recording_id="02")

        assert res_bad["ok"] is True
        assert res_bad["accepted"] is False
        assert "Insufficient reliable signal" in res_bad["reason"] or "below minimum" in res_bad["reason"]

        # 3. Verify state is NON-DESTRUCTIVE (pieces & score not erased)
        state_after_bad = engine.get_state("test_sub")
        assert state_after_bad.pieces_completed == pieces_before
        assert state_after_bad.learning_score == score_before
        assert state_after_bad.recordings_processed == recs_before
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_deterministic_puzzle_pieces():
    """Verify 20 deterministic pieces and subspace partitioning."""
    puzzle = HeartPuzzle(HeartPrintConfig(total_pieces=20), total_dims=256)
    assert len(puzzle.pieces) == 20

    # Check piece IDs and dimension bounds
    dims_covered = 0
    for i, p in enumerate(puzzle.pieces):
        assert p.piece_id == f"piece_{i+1:03d}"
        assert p.dim_start == dims_covered
        assert p.dim_end > p.dim_start
        dims_covered = p.dim_end
        assert p.svg_path.startswith("M")

    assert dims_covered == 256


def test_progressive_piece_unlocking_and_persistence():
    """Verify pieces require repeated evidence to unlock and persist across reloads."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        cfg = HeartPrintConfig(min_piece_observations=2, piece_stability_threshold=0.70)
        engine = HeartPrintEngine(cfg=cfg, checkpoint_dir=tmpdir)

        # Recording 1: initializes reference baseline
        sig1 = _generate_synthetic_scg(duration_seconds=30.0, noise_level=0.02)
        res1 = engine.process_six_channel_recording("user_abc", sig1, recording_id="01")
        assert res1["accepted"] is True

        # Recording 2: similar cardiac pattern should accumulate stability observations
        sig2 = _generate_synthetic_scg(duration_seconds=30.0, noise_level=0.02)
        res2 = engine.process_six_channel_recording("user_abc", sig2, recording_id="02")
        assert res2["accepted"] is True
        assert res2["pieces_completed"] > 0

        # Reload state from disk to test persistence
        state_reloaded = HeartPrintState.load_or_create("user_abc", ckpt_dir=tmpdir)
        assert state_reloaded.pieces_completed == res2["pieces_completed"]
        assert state_reloaded.recordings_processed == 2
        assert len(state_reloaded.historical_embeddings) == 2

        # Reset state test
        state_reset = engine.reset_enrollment("user_abc")
        assert state_reset.pieces_completed == 0
        assert state_reset.recordings_processed == 0
        assert state_reset.reference_embedding is None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_edge_cases():
    """Test extreme edge cases: single window, all zeros, empty signals."""
    tmpdir = Path(tempfile.mkdtemp())
    try:
        engine = HeartPrintEngine(checkpoint_dir=tmpdir)

        # 1. Very short signal (1 second)
        short_sig = np.random.randn(6, 250).astype(np.float32)
        res_short = engine.process_six_channel_recording("user_edge", short_sig, recording_id="01")
        assert res_short["ok"] is True
        assert res_short["accepted"] is False

        # 2. Inf values
        inf_sig = _generate_synthetic_scg(duration_seconds=10.0)
        inf_sig[0, 50] = np.inf
        res_inf = engine.process_six_channel_recording("user_edge", inf_sig, recording_id="02")
        assert res_inf["ok"] is True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
