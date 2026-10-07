"""
src/heartprint/state.py

Data model and persistence for HeartPrint progressive enrollment.
Stores subject-specific enrollment progress, piece states, learning scores,
and historical embedding representations with atomic persistence.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CHECKPOINT_DIR, HeartPrintConfig, HEARTPRINT_CFG


@dataclass
class HeartPrintState:
    subject_id: str
    heartprint_version: int = 1
    pieces_completed: int = 0
    total_pieces: int = 20
    learning_score: float = 0.0
    signal_quality: float = 0.0
    embedding_stability: float = 0.0
    reconstruction_quality: float = 0.0
    cross_session_similarity: float = 0.0
    verification_consistency: float = 0.0
    heart_twin_established: bool = False
    sessions_enrolled: int = 0
    recordings_processed: int = 0
    piece_states: dict[str, dict] = field(default_factory=dict)
    reference_embedding: list[float] | None = None
    historical_embeddings: list[list[float]] = field(default_factory=list)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_recording_summary: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def state_path_for_user(cls, subject_id: str, ckpt_dir: Path = CHECKPOINT_DIR) -> Path:
        return ckpt_dir / f"heartprint_user_{subject_id}.json"

    @classmethod
    def load_or_create(cls, subject_id: str, ckpt_dir: Path = CHECKPOINT_DIR, cfg: HeartPrintConfig | None = None) -> "HeartPrintState":
        cfg = cfg or HEARTPRINT_CFG
        path = cls.state_path_for_user(subject_id, ckpt_dir)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(
                    subject_id=data.get("subject_id", subject_id),
                    heartprint_version=data.get("heartprint_version", 1),
                    pieces_completed=data.get("pieces_completed", 0),
                    total_pieces=data.get("total_pieces", cfg.total_pieces),
                    learning_score=data.get("learning_score", 0.0),
                    signal_quality=data.get("signal_quality", 0.0),
                    embedding_stability=data.get("embedding_stability", 0.0),
                    reconstruction_quality=data.get("reconstruction_quality", 0.0),
                    cross_session_similarity=data.get("cross_session_similarity", 0.0),
                    verification_consistency=data.get("verification_consistency", 0.0),
                    heart_twin_established=data.get("heart_twin_established", False),
                    sessions_enrolled=data.get("sessions_enrolled", 0),
                    recordings_processed=data.get("recordings_processed", 0),
                    piece_states=data.get("piece_states", {}),
                    reference_embedding=data.get("reference_embedding", None),
                    historical_embeddings=data.get("historical_embeddings", []),
                    last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
                    last_recording_summary=data.get("last_recording_summary", {}),
                )
            except Exception as e:
                print(f"[HeartPrintState] Warning: Could not read {path}: {e}; creating fresh state.")

        # Initialize fresh state
        initial_pieces = {}
        for i in range(1, cfg.total_pieces + 1):
            pid = f"piece_{i:03d}"
            initial_pieces[pid] = {
                "established": False,
                "stability": 0.0,
                "observations": 0,
                "just_unlocked": False,
            }

        state = cls(
            subject_id=subject_id,
            total_pieces=cfg.total_pieces,
            piece_states=initial_pieces,
        )
        return state

    def save(self, ckpt_dir: Path = CHECKPOINT_DIR) -> Path:
        """Atomically saves state to disk."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_path_for_user(self.subject_id, ckpt_dir)
        temp_path = path.with_suffix(".tmp")
        
        self.last_updated = datetime.now(timezone.utc).isoformat()
        payload = asdict(self)
        
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
        return path

    def reset(self, ckpt_dir: Path = CHECKPOINT_DIR, cfg: HeartPrintConfig | None = None) -> "HeartPrintState":
        """Resets enrollment for this subject and saves to disk."""
        cfg = cfg or HEARTPRINT_CFG
        self.pieces_completed = 0
        self.learning_score = 0.0
        self.signal_quality = 0.0
        self.embedding_stability = 0.0
        self.reconstruction_quality = 0.0
        self.cross_session_similarity = 0.0
        self.verification_consistency = 0.0
        self.heart_twin_established = False
        self.sessions_enrolled = 0
        self.recordings_processed = 0
        self.reference_embedding = None
        self.historical_embeddings = []
        self.last_recording_summary = {"status": "Enrollment reset."}

        initial_pieces = {}
        for i in range(1, cfg.total_pieces + 1):
            pid = f"piece_{i:03d}"
            initial_pieces[pid] = {
                "established": False,
                "stability": 0.0,
                "observations": 0,
                "just_unlocked": False,
            }
        self.piece_states = initial_pieces
        self.save(ckpt_dir)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Returns clean serializable dictionary for JSON API responses."""
        return asdict(self)
