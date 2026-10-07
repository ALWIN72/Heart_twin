"""
tests/test_pipeline.py

End-to-end smoke tests for every phase of the Digital Heart Twin Digital Heart Twin
pipeline. These are fast (small synthetic data, tiny models/epoch counts)
and meant to catch shape/wiring regressions, not to validate model quality.

Run with:
    python -m pytest tests/test_pipeline.py -v
or:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TwinConfig
from src.data_loader import build_multimodal_manifest, create_splits
from src.dataset import SCGWindowDataset
from src.models.dann import DeviceHarmonizer
from src.models.denoiser import MotionArtifactCanceller
from src.models.encoder import SCGEncoder
from src.models.mae import MaskedAutoencoder
from src.models.personal_twin import PersonalHeartTwin
from src.models.placement_classifier import PlacementClassifier
from src.models.spatial_transformer import SpatialCorrector
from src.utils.synthetic_data import generate_synthetic_dataset


def _make_temp_dataset(n_subjects=4, recordings_per_subject=4, duration_seconds=12):
    tmpdir = Path(tempfile.mkdtemp())
    generate_synthetic_dataset(out_root=tmpdir, n_subjects=n_subjects,
                                recordings_per_subject=recordings_per_subject,
                                duration_seconds=duration_seconds, seed=1)
    return tmpdir


def test_data_pipeline():
    tmpdir = _make_temp_dataset()
    try:
        manifest = build_multimodal_manifest(raw_data_path=tmpdir)
        assert len(manifest) > 0, "manifest should not be empty"
        assert set(["subject_id", "recording_id", "cal_scg"]).issubset(manifest.columns)

        train_df, val_df, test_df = create_splits(
            manifest_path=Path("data/manifests/multimodal_manifest.csv"))
        # no subject should appear in more than one split (leakage check)
        assert set(train_df.subject_id).isdisjoint(set(val_df.subject_id))
        assert set(train_df.subject_id).isdisjoint(set(test_df.subject_id))
        assert set(val_df.subject_id).isdisjoint(set(test_df.subject_id))

        ds = SCGWindowDataset(manifest)
        assert len(ds) > 0
        x, meta = ds[0]
        assert x.shape[0] == 6, "expected 6 channels (3 SCG + 3 gyro/gyro-proxy)"
        assert "subject_id" in meta
        print("test_data_pipeline: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_encoder_shapes():
    cfg = ModelConfig()
    enc = SCGEncoder(cfg)
    x = torch.randn(2, 6, 2500)
    tokens = enc(x)
    assert tokens.shape == (2, cfg.num_patches, cfg.embed_dim)
    pooled = enc.encode_pooled(x)
    assert pooled.shape == (2, cfg.embed_dim)
    print("test_encoder_shapes: PASS")


def test_mae_forward_backward():
    cfg = ModelConfig()
    mae = MaskedAutoencoder(cfg)
    x = torch.randn(2, 6, 2500)
    out = mae(x)
    assert out["loss"].item() >= 0
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert mae.patch_embed.proj.weight.grad is not None
    encoder = mae.export_encoder()
    tokens = encoder(x)
    assert tokens.shape == (2, cfg.num_patches, cfg.embed_dim)
    print("test_mae_forward_backward: PASS")


def test_denoiser():
    cfg = ModelConfig()
    foundation = SCGEncoder(cfg)
    canceller = MotionArtifactCanceller(foundation, cfg)
    x = torch.randn(2, 6, 2500)
    patches = canceller(x)
    assert patches.shape == (2, cfg.num_patches, 3 * cfg.patch_size)
    waveform = canceller.reconstruct_waveform(x)
    assert waveform.shape == (2, 3, 2500)
    # confirm early layers are actually frozen
    frozen_params = [p for i, b in enumerate(canceller.encoder.blocks) if i < 2 for p in b.parameters()]
    assert all(not p.requires_grad for p in frozen_params)
    print("test_denoiser: PASS")


def test_harmonizer_dann():
    cfg = ModelConfig()
    enc = SCGEncoder(cfg)
    harmonizer = DeviceHarmonizer(enc, num_devices=2, cfg=cfg)
    x = torch.randn(2, 6, 2500)
    recon, domain_pred = harmonizer(x, alpha=1.0)
    assert recon.shape == (2, cfg.num_patches, cfg.in_channels * cfg.patch_size)
    assert domain_pred.shape == (2, 2)
    harmonized = harmonizer.harmonize(x)
    assert harmonized.shape == x.shape
    print("test_harmonizer_dann: PASS")


def test_placement_and_spatial_corrector():
    clf = PlacementClassifier()
    x = torch.randn(3, 6, 2500)
    logits = clf(x)
    assert logits.shape == (3, 3)

    corrector = SpatialCorrector()
    probs = torch.softmax(logits, dim=-1)
    corrected = corrector(x, probs)
    assert corrected.shape == x.shape

    hard_idx = torch.tensor([0, 1, 2])
    corrected_hard = corrector.forward_hard_label(x, hard_idx)
    assert corrected_hard.shape == x.shape
    print("test_placement_and_spatial_corrector: PASS")


def test_personal_twin_anomaly_detection():
    cfg = ModelConfig()
    twin_cfg = TwinConfig()
    foundation = SCGEncoder(cfg)
    twin = PersonalHeartTwin(foundation, user_id="test_user", model_cfg=cfg, twin_cfg=twin_cfg)

    baseline = torch.randn(8, 6, 2500) * 0.3
    twin.set_baseline(baseline)
    assert bool(twin.has_baseline.item())

    similar = torch.randn(4, 6, 2500) * 0.3
    different = torch.randn(4, 6, 2500) * 8.0 + 15.0

    scores_similar = twin.anomaly_score(similar)
    scores_different = twin.anomaly_score(different)

    assert scores_similar.shape == (4,)
    assert (scores_different.mean() > scores_similar.mean()), \
        "anomaly score should be higher for out-of-distribution input"

    # global encoder must remain frozen through training
    out = twin(baseline)
    loss = ((out["recon"] - out["target"]) ** 2).mean()
    loss.backward()
    assert foundation.blocks[0].norm1.weight.grad is None, "foundation encoder should stay frozen"
    print("test_personal_twin_anomaly_detection: PASS")


def test_personal_twin_requires_baseline():
    cfg = ModelConfig()
    foundation = SCGEncoder(cfg)
    twin = PersonalHeartTwin(foundation, user_id="no_baseline_user")
    x = torch.randn(2, 6, 2500)
    try:
        twin.anomaly_score(x)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "anomaly_score should raise RuntimeError before set_baseline() is called"
    print("test_personal_twin_requires_baseline: PASS")


def run_all():
    test_data_pipeline()
    test_encoder_shapes()
    test_mae_forward_backward()
    test_denoiser()
    test_harmonizer_dann()
    test_placement_and_spatial_corrector()
    test_personal_twin_anomaly_detection()
    test_personal_twin_requires_baseline()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
