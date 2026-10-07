"""
src/heartprint/engine.py

Core HeartPrint Engine: Orchestrates signal windowing, quality gating,
neural embedding generation, quality-weighted aggregation, representation
stability evaluation, puzzle piece unlocking, and non-destructive state persistence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    CHECKPOINT_DIR,
    SAMPLING_RATE_HZ,
    WINDOW_SAMPLES,
    HeartPrintConfig,
    ModelConfig,
    TwinConfig,
    HEARTPRINT_CFG,
    get_device,
)
from src.dataset import _load_scg_csv
from src.heartprint.puzzle import HeartPuzzle
from src.heartprint.quality import WindowQualityAssessor
from src.heartprint.state import HeartPrintState
from src.models.encoder import SCGEncoder
from src.models.personal_twin import PersonalHeartTwin
from src.utils.preprocessing import (
    assemble_six_channel,
    make_fixed_windows,
    normalize_per_channel,
    resample_signal,
)

logger = logging.getLogger("HeartPrint")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


class HeartPrintEngine:
    """
    Main engine for Progressive Cardiac Identity Enrollment.
    """

    def __init__(
        self,
        cfg: HeartPrintConfig | None = None,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        device: torch.device | None = None,
    ):
        self.cfg = cfg or HEARTPRINT_CFG
        self.checkpoint_dir = checkpoint_dir
        self.device = device or get_device()
        self.quality_assessor = WindowQualityAssessor(self.cfg, fs=SAMPLING_RATE_HZ)
        self.puzzle = HeartPuzzle(self.cfg, total_dims=256)

        # Foundation model
        self.encoder_cfg = ModelConfig()
        self.foundation_encoder = self._load_foundation_encoder()

    def _load_foundation_encoder(self) -> SCGEncoder:
        """Loads foundation encoder checkpoint or initializes if not yet trained."""
        encoder = SCGEncoder(self.encoder_cfg)
        ckpt_path = self.checkpoint_dir / "encoder_foundation.pt"
        if not ckpt_path.exists():
            ckpt_path = self.checkpoint_dir / "encoder_real.pt"

        if ckpt_path.exists():
            try:
                encoder.load_state_dict(
                    torch.load(ckpt_path, map_location="cpu", weights_only=True)
                )
                logger.info(f"Loaded foundation encoder from {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load foundation encoder from {ckpt_path}: {e}")
        else:
            logger.info("Foundation checkpoint not found; running with initialized SCGEncoder.")

        return encoder.to(self.device).eval()

    def _load_user_twin(self, subject_id: str) -> PersonalHeartTwin | None:
        """Loads personal twin model for subject if available."""
        ckpt_path = self.checkpoint_dir / f"twin_user_{subject_id}.pt"
        if not ckpt_path.exists():
            return None
        try:
            twin = PersonalHeartTwin(
                self.foundation_encoder,
                user_id=subject_id,
                model_cfg=self.encoder_cfg,
                twin_cfg=TwinConfig(),
            )
            twin.load_state_dict(
                torch.load(ckpt_path, map_location="cpu", weights_only=True),
                strict=False,
            )
            return twin.to(self.device).eval()
        except Exception as e:
            logger.warning(f"Could not load personal twin for subject {subject_id}: {e}")
            return None

    def get_state(self, subject_id: str) -> HeartPrintState:
        """Retrieves or initializes HeartPrintState for a subject."""
        return HeartPrintState.load_or_create(
            subject_id=subject_id, ckpt_dir=self.checkpoint_dir, cfg=self.cfg
        )

    def reset_enrollment(self, subject_id: str) -> HeartPrintState:
        """Resets enrollment for a subject."""
        state = self.get_state(subject_id)
        state.reset(self.checkpoint_dir, self.cfg)
        logger.info(f"Subject {subject_id} HeartPrint enrollment reset.")
        return state

    def process_six_channel_recording(
        self,
        subject_id: str,
        six_channel_signal: np.ndarray,
        recording_id: str = "01",
        detected_bpm: float | None = None,
    ) -> dict[str, Any]:
        """
        Processes a raw (6, T) physiological signal array through the progressive enrollment pipeline.
        """
        logger.info(f"Recording received for Subject {subject_id} (Recording {recording_id})")
        state = self.get_state(subject_id)

        # 1. Windowing
        window_len_samples = int(self.cfg.window_seconds * SAMPLING_RATE_HZ)
        stride_samples = int(self.cfg.window_stride_seconds * SAMPLING_RATE_HZ)

        windows = make_fixed_windows(
            six_channel_signal,
            window_samples=window_len_samples,
            stride=stride_samples,
        )
        n_windows = len(windows)
        logger.info(f"Generated {n_windows} windows (length={self.cfg.window_seconds}s, stride={self.cfg.window_stride_seconds}s)")

        # 2. Window-level Signal Quality Assessment
        window_sqis = [self.quality_assessor.compute_window_sqi(w) for w in windows]
        sqi_scores = [w["sqi"] for w in window_sqis]
        viability = self.quality_assessor.evaluate_recording_viability(window_sqis)

        usable_count = viability["usable_count"]
        med_quality = viability["median_quality"]
        logger.info(f"Windows: {n_windows} | Usable windows: {usable_count}/{n_windows} | Median quality: {med_quality:.2f}")

        # 3. Quality gating: If insufficient reliable signal, abstain and preserve state
        if not viability["viable"]:
            logger.warning(
                f"Recording rejected for enrollment: {viability['reason']} | Heart Twin unchanged"
            )
            state.last_recording_summary = {
                "recording_id": recording_id,
                "viable": False,
                "usable_windows": f"{usable_count}/{n_windows}",
                "median_quality": med_quality,
                "reason": viability["reason"],
                "window_sqis": window_sqis,
            }
            state.save(self.checkpoint_dir)

            return {
                "ok": True,
                "accepted": False,
                "subject_id": subject_id,
                "recording_id": recording_id,
                "reason": viability["reason"],
                "viable": False,
                "usable_windows": usable_count,
                "total_windows": n_windows,
                "usable_ratio": viability["usable_ratio"],
                "median_quality": med_quality,
                "window_sqis": window_sqis,
                "heartprint_state": state.to_dict(),
                "puzzle_pieces": self.puzzle.pieces,
                "detected_bpm": detected_bpm,
            }

        # 4. Neural Encoder: Extract embeddings for usable windows
        usable_indices = [i for i, w in enumerate(window_sqis) if w["usable"]]
        usable_windows_arr = windows[usable_indices]  # (B_usable, 6, window_samples)
        usable_sqis = [sqi_scores[i] for i in usable_indices]
        weights = self.quality_assessor.compute_quality_weights(usable_sqis)

        # Prepare normalized tensor for encoder
        usable_tensors = []
        for w in usable_windows_arr:
            norm_w = normalize_per_channel(w)
            usable_tensors.append(torch.from_numpy(norm_w).float())

        x_batch = torch.stack(usable_tensors, dim=0).to(self.device)

        with torch.no_grad():
            # Foundation encoder pooled features (B_usable, 256)
            pooled_feats = self.foundation_encoder.encode_pooled(x_batch).cpu().numpy()

        # 5. Quality-Weighted Embedding Aggregation
        # Numerical stability: sum(w_i * z_i) / (sum(w_i) + eps)
        weights_expanded = weights[:, None]
        weight_sum = float(np.sum(weights)) + 1e-9
        weighted_embedding = np.sum(pooled_feats * weights_expanded, axis=0) / weight_sum
        
        # Unit-normalize aggregated embedding vector
        norm_val = np.linalg.norm(weighted_embedding) + 1e-9
        weighted_embedding = (weighted_embedding / norm_val).astype(np.float64)
        logger.info("Quality-weighted aggregate embedding generated")

        # 6. Personal Twin Pass & Reconstruction Metric
        twin = self._load_user_twin(subject_id)
        recon_quality = 0.85  # Default solid score when twin is in learning phase
        verification_consistency = 0.85

        if twin is not None:
            try:
                with torch.no_grad():
                    z_tensor = torch.from_numpy(weighted_embedding[None, :]).float().to(self.device)
                    # Pass through personal encoder and decoder
                    personal_hidden = twin.personal_encoder(z_tensor)
                    mu = twin.fc_mu(personal_hidden)
                    logvar = twin.fc_logvar(personal_hidden)
                    z_latent = twin.reparameterize(mu, logvar)
                    recon_feat = twin.decoder(z_latent)
                    
                    # Reconstruction MSE
                    recon_mse = float(F.mse_loss(recon_feat, z_tensor).item())
                    # Normalized reconstruction quality in [0, 1]
                    recon_quality = float(np.clip(1.0 / (1.0 + recon_mse * 2.0), 0.0, 1.0))

                    if bool(twin.has_baseline.item()):
                        kl_div = float(twin.anomaly_score(x_batch[:1]).item())
                        verification_consistency = float(np.clip(np.exp(-kl_div / 2.0), 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Twin scoring exception: {e}")

        # 7. Representation Stability & Cross-Session Similarity
        ref_emb = (
            np.array(state.reference_embedding, dtype=np.float64)
            if state.reference_embedding is not None
            else None
        )

        if ref_emb is not None:
            # Cosine similarity against reference template
            sim_ref = float(
                np.dot(weighted_embedding, ref_emb)
                / ((np.linalg.norm(weighted_embedding) * np.linalg.norm(ref_emb)) + 1e-9)
            )
            sim_ref = float(np.clip(sim_ref, 0.0, 1.0))

            # Cross-session consistency against historical embeddings
            if len(state.historical_embeddings) > 0:
                hist_sims = []
                for h_emb in state.historical_embeddings[-5:]:
                    h_arr = np.array(h_emb, dtype=np.float64)
                    s = np.dot(weighted_embedding, h_arr) / (
                        (np.linalg.norm(weighted_embedding) * np.linalg.norm(h_arr)) + 1e-9
                    )
                    hist_sims.append(float(np.clip(s, 0.0, 1.0)))
                cross_session_similarity = float(np.mean(hist_sims))
            else:
                cross_session_similarity = sim_ref

            embedding_stability = float(
                np.clip(0.6 * sim_ref + 0.4 * cross_session_similarity, 0.0, 1.0)
            )

            # Update reference template smoothly with exponential moving average (alpha=0.75)
            alpha = 0.75
            updated_ref = alpha * ref_emb + (1 - alpha) * weighted_embedding
            updated_ref = updated_ref / (np.linalg.norm(updated_ref) + 1e-9)
            state.reference_embedding = updated_ref.tolist()
        else:
            # First enrollment recording initializes reference
            embedding_stability = 0.78
            cross_session_similarity = 0.75
            state.reference_embedding = weighted_embedding.tolist()

        logger.info(
            f"Embedding stability: {embedding_stability:.2f} | Cross-session similarity: {cross_session_similarity:.2f}"
        )

        # Append to historical representations (keep last 12)
        state.historical_embeddings.append(weighted_embedding.tolist())
        if len(state.historical_embeddings) > 12:
            state.historical_embeddings.pop(0)

        # 8. Evaluate 20-Piece Subspace Stability
        piece_results = self.puzzle.evaluate_piece_stability(
            current_embedding=weighted_embedding,
            reference_embedding=ref_emb,
            existing_piece_states=state.piece_states,
        )
        state.piece_states = piece_results

        established_count = sum(1 for p in piece_results.values() if p["established"])
        logger.info(f"Established pieces: {established_count}/{self.cfg.total_pieces}")

        # 9. Calculate Composite Learning Score
        w1 = self.cfg.w_signal_quality
        w2 = self.cfg.w_embedding_stability
        w3 = self.cfg.w_reconstruction_quality
        w4 = self.cfg.w_session_consistency
        w5 = self.cfg.w_verification

        learning_score = (
            w1 * med_quality
            + w2 * embedding_stability
            + w3 * recon_quality
            + w4 * cross_session_similarity
            + w5 * verification_consistency
        )
        learning_score = float(np.clip(learning_score, 0.0, 1.0))

        # 10. Update Enrollment State
        state.pieces_completed = established_count
        state.learning_score = round(learning_score, 4)
        state.signal_quality = round(med_quality, 4)
        state.embedding_stability = round(embedding_stability, 4)
        state.reconstruction_quality = round(recon_quality, 4)
        state.cross_session_similarity = round(cross_session_similarity, 4)
        state.verification_consistency = round(verification_consistency, 4)
        state.recordings_processed += 1
        state.sessions_enrolled += 1

        # Check full enrollment condition
        if established_count >= self.cfg.total_pieces and learning_score >= 0.78:
            state.heart_twin_established = True
            logger.info("HEARTPRINT ESTABLISHED: Cardiac representation successfully enrolled.")

        state.last_recording_summary = {
            "recording_id": recording_id,
            "viable": True,
            "usable_windows": f"{usable_count}/{n_windows}",
            "median_quality": med_quality,
            "embedding_stability": embedding_stability,
            "pieces_completed": established_count,
            "learning_score": learning_score,
            "heart_twin_established": state.heart_twin_established,
            "window_sqis": window_sqis,
        }

        # Persist non-destructive state
        state.save(self.checkpoint_dir)

        return {
            "ok": True,
            "accepted": True,
            "subject_id": subject_id,
            "recording_id": recording_id,
            "viable": True,
            "usable_windows": usable_count,
            "total_windows": n_windows,
            "usable_ratio": viability["usable_ratio"],
            "median_quality": round(med_quality, 4),
            "embedding_stability": round(embedding_stability, 4),
            "reconstruction_quality": round(recon_quality, 4),
            "cross_session_similarity": round(cross_session_similarity, 4),
            "learning_score": round(learning_score, 4),
            "pieces_completed": established_count,
            "total_pieces": self.cfg.total_pieces,
            "heart_twin_established": state.heart_twin_established,
            "window_sqis": window_sqis,
            "piece_states": piece_results,
            "heartprint_state": state.to_dict(),
            "detected_bpm": detected_bpm,
        }

    def process_recording(
        self,
        recording_row: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convenience method to process a recording directly from a manifest dictionary or DataFrame row.
        """
        sub_id = str(recording_row.get("subject_id", "unknown"))
        rec_id = str(recording_row.get("recording_id", "01"))

        cal_path = recording_row.get("cal_scg")
        uncal_path = recording_row.get("uncal_scg")
        gyro_path = recording_row.get("gyro")

        if not cal_path or not Path(str(cal_path)).exists():
            return {
                "ok": False,
                "error": f"Signal file not found at {cal_path}",
            }

        cal_raw, fs_cal = _load_scg_csv(str(cal_path))
        cal = resample_signal(cal_raw, fs_cal, SAMPLING_RATE_HZ)

        if uncal_path and isinstance(uncal_path, str) and Path(uncal_path).exists():
            uncal_raw, fs_uncal = _load_scg_csv(str(uncal_path))
            uncal = resample_signal(uncal_raw, fs_uncal, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], uncal.shape[-1])
            cal, uncal = cal[:, :min_len], uncal[:, :min_len]
        else:
            uncal = cal.copy()

        gyro_real = None
        if gyro_path and isinstance(gyro_path, str) and Path(gyro_path).exists():
            gyro_raw, fs_gyro = _load_scg_csv(str(gyro_path))
            gyro_real = resample_signal(gyro_raw, fs_gyro, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], gyro_real.shape[-1])
            gyro_real = gyro_real[:, :min_len]
            cal = cal[:, :min_len]
            uncal = uncal[:, :min_len]

        six_ch = assemble_six_channel(cal, uncal, gyro=gyro_real, fs=SAMPLING_RATE_HZ)
        return self.process_six_channel_recording(
            subject_id=sub_id,
            six_channel_signal=six_ch,
            recording_id=rec_id,
        )

    def process_saved_recording(
        self,
        subject_id: str,
        recording_id: str = "01",
        raw_root: Path = Path("."),
    ) -> dict[str, Any]:
        """
        Loads a saved recording from disk and processes it for HeartPrint enrollment.
        """
        # Try both 3-digit and 4-digit formatting conventions
        candidates = [
            raw_root / "MSCardio" / f"Subject_{subject_id}" / f"Recording_{recording_id}",
            raw_root / "MSCardio" / f"Subject_{subject_id.zfill(4)}" / f"Recording_{recording_id.zfill(3)}",
            raw_root / "MSCardio" / f"Subject_{subject_id.zfill(3)}" / f"Recording_{recording_id.zfill(2)}",
        ]
        rec_dir = next((c for c in candidates if c.exists()), candidates[0])
        cal_path = rec_dir / "scg.csv"
        uncal_path = rec_dir / "Uncalibrated_scg.csv"
        gyro_path = rec_dir / "gyro.csv"

        if not cal_path.exists():
            return {
                "ok": False,
                "error": f"Recording not found at {cal_path}",
            }

        cal_raw, fs_cal = _load_scg_csv(str(cal_path))
        cal = resample_signal(cal_raw, fs_cal, SAMPLING_RATE_HZ)

        if uncal_path.exists():
            uncal_raw, fs_uncal = _load_scg_csv(str(uncal_path))
            uncal = resample_signal(uncal_raw, fs_uncal, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], uncal.shape[-1])
            cal, uncal = cal[:, :min_len], uncal[:, :min_len]
        else:
            uncal = cal.copy()

        gyro_real = None
        if gyro_path.exists():
            gyro_raw, fs_gyro = _load_scg_csv(str(gyro_path))
            gyro_real = resample_signal(gyro_raw, fs_gyro, SAMPLING_RATE_HZ)
            min_len = min(cal.shape[-1], gyro_real.shape[-1])
            gyro_real = gyro_real[:, :min_len]
            cal = cal[:, :min_len]
            uncal = uncal[:, :min_len]

        six_ch = assemble_six_channel(cal, uncal, gyro=gyro_real, fs=SAMPLING_RATE_HZ)
        return self.process_six_channel_recording(
            subject_id=subject_id,
            six_channel_signal=six_ch,
            recording_id=recording_id,
        )

