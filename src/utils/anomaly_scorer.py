"""
src/utils/anomaly_scorer.py

Phase 4.2: Production Prediction Pipeline.

`HealthMonitor` wires together every model trained in Phases 1-4 into a
single per-user inference pipeline:

    raw recording
        -> PlacementClassifier + SpatialCorrector  (normalize sensor orientation)
        -> DeviceHarmonizer                         (normalize across phones)
        -> PersonalHeartTwin.anomaly_score           (compare to this user's own history)
        -> trend analysis + human-readable alert

This replaces the original sketch's undefined globals (placement_classifier,
spatial_corrector, device_harmonizer used as free functions) with an
explicit, loadable model bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, ModelConfig, TwinConfig, get_device
from src.models.dann import DeviceHarmonizer
from src.models.encoder import SCGEncoder
from src.models.personal_twin import PersonalHeartTwin
from src.models.placement_classifier import PlacementClassifier
from src.models.spatial_transformer import SpatialCorrector


def load_model_bundle(checkpoint_dir: Path = CHECKPOINT_DIR, device: torch.device | None = None):
    """
    Loads every shared (non-per-user) model needed by HealthMonitor:
    foundation encoder, placement classifier, spatial corrector, device
    harmonizer. Missing checkpoints fall back to randomly-initialized
    modules with a warning, so the pipeline is still runnable (with
    degraded quality) before every phase has been trained.
    """
    device = device or get_device()
    cfg = ModelConfig()

    def _load(model, ckpt_name, label):
        path = checkpoint_dir / ckpt_name
        if path.exists():
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            print(f"[HealthMonitor] WARNING: {path} not found; using randomly-initialized {label}.")
        return model.to(device).eval()

    foundation = _load(SCGEncoder(cfg), "encoder_foundation.pt", "foundation encoder")
    placement_clf = _load(PlacementClassifier(), "placement_classifier.pt", "placement classifier")
    spatial_corrector = SpatialCorrector().to(device).eval()  # trained jointly w/ placement in some setups; identity-ish by default
    harmonizer = _load(DeviceHarmonizer(SCGEncoder(cfg), num_devices=2, cfg=cfg), "harmonizer.pt", "device harmonizer")

    return {
        "foundation": foundation,
        "placement_clf": placement_clf,
        "spatial_corrector": spatial_corrector,
        "harmonizer": harmonizer,
        "device": device,
        "cfg": cfg,
    }


def load_user_twin(user_id: str, foundation: SCGEncoder, checkpoint_dir: Path = CHECKPOINT_DIR,
                    device: torch.device | None = None) -> PersonalHeartTwin | None:
    device = device or get_device()
    cfg = ModelConfig()
    twin_cfg = TwinConfig()
    path = checkpoint_dir / f"twin_user_{user_id}.pt"
    if not path.exists():
        return None
    twin = PersonalHeartTwin(foundation, user_id=user_id, model_cfg=cfg, twin_cfg=twin_cfg)
    twin.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    return twin.to(device).eval()



@dataclass
class HealthMonitor:
    """
    Per-user runtime monitor. Keeps a rolling history of anomaly scores so
    trend/lead-time analysis (Phase 5/7) can be computed incrementally as
    new recordings arrive.
    """
    user_id: str
    bundle: dict
    twin: PersonalHeartTwin | None = None
    history: list[float] = field(default_factory=list)
    threshold: float = TwinConfig().anomaly_threshold

    @classmethod
    def for_user(cls, user_id: str, checkpoint_dir: Path = CHECKPOINT_DIR, device: torch.device | None = None) -> "HealthMonitor":
        bundle = load_model_bundle(checkpoint_dir, device)
        twin = load_user_twin(user_id, bundle["foundation"], checkpoint_dir, bundle["device"])
        return cls(user_id=user_id, bundle=bundle, twin=twin)

    @torch.no_grad()
    def analyze_recording(self, new_signal: torch.Tensor) -> dict:
        """
        new_signal: (1, 6, T) or (6, T) raw windowed 6-channel signal for
        a single recording (already resampled/normalized via
        src.utils.preprocessing -- see dataset.py's SCGWindowDataset for
        the canonical preprocessing path).
        """
        if self.twin is None:
            return {"error": f"No trained PersonalHeartTwin found for user {self.user_id}. "
                              f"Run train_personal_twin.py for this user first."}

        if new_signal.dim() == 2:
            new_signal = new_signal.unsqueeze(0)
        device = self.bundle["device"]
        new_signal = new_signal.to(device)

        # 1. Placement detection + correction
        placement_logits = self.bundle["placement_clf"](new_signal)
        placement_probs = F.softmax(placement_logits, dim=-1)
        placement_label_idx = int(placement_probs.argmax(dim=-1).item())
        signal_corrected = self.bundle["spatial_corrector"](new_signal, placement_probs)

        # 2. Device harmonization
        signal_normalized = self.bundle["harmonizer"].harmonize(signal_corrected)

        # 3. Pass through this user's Digital Twin
        kl_score = self.twin.anomaly_score(signal_normalized)  # (1,)
        score_val = float(kl_score.item())

        # 4. Update history + compute trend
        self.history.append(score_val)
        result = {"score": score_val, "placement": placement_label_idx}

        if len(self.history) > 10:
            x_axis = np.arange(len(self.history))
            trend = float(np.polyfit(x_axis, self.history, 1)[0])
            baseline_ref = self.history[0] if abs(self.history[0]) > 1e-8 else 1e-8
            change_percent = (self.history[-1] - self.history[0]) / abs(baseline_ref) * 100

            if score_val > self.threshold:
                alert = (f"WARNING: Cardiac pattern deviated by {change_percent:.1f}% from baseline. "
                         f"Risk of deterioration increased.")
            else:
                alert = f"Stable. Current deviation: {change_percent:.1f}%."

            result.update({"trend": trend, "alert": alert, "change_percent": change_percent})
        else:
            result["status"] = f"Building baseline history ({len(self.history)}/11 recordings)..."

        return result

    def reset_history(self):
        self.history = []
