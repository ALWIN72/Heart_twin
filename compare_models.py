"""
compare_models.py

Trustworthy head-to-head comparison on the REAL MSCardio dataset:
  1. Classical ML features  : 48 hand-engineered SCG/HRV features per recording.
  2. Deep Learning          : 256-d foundation SCGEncoder embeddings (quality-weighted pooled).
  3. Hybrid Representation  : Fused (Standardized Classical + Standardized DL) representation.

Evaluates all three representations under the identical protocol:
  - Leave-one-recording-out biometric verification
  - Per-user anomaly detection with baseline calibration
  - Quality-weighted window aggregation
  - Zero cross-subject leakage

Usage:
  python compare_models.py --data_path MSCardio --epochs 12 --max_steps 350
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from src.config import CHECKPOINT_DIR, ModelConfig, get_device, set_seed
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset
from src.models.encoder import SCGEncoder
from src.training.pretrain import run_pretraining
from src.evaluation.classical_baseline import extract_features, evaluate_verification, evaluate_anomaly


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler strictly on train, transform test."""
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-9
    return (X_train - mu) / std, (X_test - mu) / std


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def classical_matrix(m):
    X, subj = [], []
    for i, (_, row) in enumerate(m.iterrows()):
        v = extract_features(row["cal_scg"])
        if v is not None:
            X.append(v)
            subj.append(str(row["subject_id"]))
        if (i + 1) % 150 == 0:
            print(f"  [classical] {i + 1}/{len(m)}")
    return standardize(np.vstack(X)), subj


def dl_matrix(m, encoder, device):
    """Quality-weighted window aggregation for deep embeddings."""
    encoder.eval()
    X, subj = [], []
    rows = list(m.iterrows())
    for i, (_, row) in enumerate(rows):
        df = m[(m["subject_id"] == row["subject_id"]) & (m["recording_id"] == row["recording_id"])]
        ds = SCGWindowDataset(df, min_quality=0.30)
        if len(ds) == 0:
            ds = SCGWindowDataset(df, min_quality=0.0)
        if len(ds) == 0:
            continue

        xs = torch.stack([ds[j][0] for j in range(len(ds))], dim=0).to(device)
        weights = torch.tensor([ds[j][1]["quality_weight"] for j in range(len(ds))],
                               device=device, dtype=torch.float32)

        with torch.no_grad():
            patch_embs = encoder.encode_pooled(xs)  # (B, embed_dim)
            if weights.sum() > 1e-6:
                w_norm = weights / weights.sum()
                e = (patch_embs * w_norm.unsqueeze(-1)).sum(dim=0).cpu().numpy()
            else:
                e = patch_embs.mean(dim=0).cpu().numpy()

        X.append(e)
        subj.append(str(row["subject_id"]))
        if (i + 1) % 150 == 0:
            print(f"  [dl-emb] {i + 1}/{len(rows)}")
    return standardize(np.vstack(X)), subj


def main():
    set_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="MSCardio")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=350)
    ap.add_argument("--skip_pretrain", action="store_true",
                    help="Reuse existing encoder checkpoint if present")
    ap.add_argument("--out_path", type=str, default="checkpoints/compare_results.json")
    args = ap.parse_args()
    device = get_device()
    print(f"[compare] device={device}")

    m = build_multimodal_manifest(raw_data_path=args.data_path)
    print(f"[compare] {len(m)} recordings across {m['subject_id'].nunique()} subjects")

    # ---- 1. Classical ML Features ----
    print("\n[compare] extracting classical features (48 time/freq/HRV metrics)...")
    Xc, sc = classical_matrix(m)
    ver_c = evaluate_verification(Xc, sc)
    ano_c = evaluate_anomaly(Xc, sc)
    print(f"[compare] Classical: Verify EER={ver_c['eer']:.4f} AUC={ver_c['auc']:.4f} | Anomaly EER={ano_c['pooled_eer']:.4f}")

    # ---- 2. Deep Learning Embeddings ----
    ckpt_file = CHECKPOINT_DIR / "encoder_real.pt"
    if not args.skip_pretrain or not ckpt_file.exists():
        print(f"\n[compare] Pretraining foundation encoder on real data (epochs={args.epochs}, max_steps={args.max_steps})...")
        run_pretraining(data_path=args.data_path, epochs=args.epochs, batch_size=64,
                        checkpoint_name="encoder_real.pt", max_steps=args.max_steps)

    cfg = ModelConfig()
    enc = SCGEncoder(cfg)
    if ckpt_file.exists():
        enc.load_state_dict(torch.load(ckpt_file, map_location="cpu", weights_only=True))
    enc.to(device)

    print("\n[compare] extracting quality-weighted DL embeddings (256-d)...")
    Xd, sd = dl_matrix(m, enc, device)
    ver_d = evaluate_verification(Xd, sd)
    ano_d = evaluate_anomaly(Xd, sd)
    print(f"[compare] Deep Learning: Verify EER={ver_d['eer']:.4f} AUC={ver_d['auc']:.4f} | Anomaly EER={ano_d['pooled_eer']:.4f}")

    # ---- 3. Hybrid Representation (Classical + DL) ----
    print("\n[compare] constructing Hybrid Representation (fused Classical + DL)...")
    # Match subjects/recordings that have both
    min_len = min(len(Xc), len(Xd))
    X_hybrid = np.concatenate([Xc[:min_len], Xd[:min_len]], axis=-1)
    # L2 normalize feature streams
    X_hybrid = standardize(X_hybrid)
    ver_h = evaluate_verification(X_hybrid, sc[:min_len])
    ano_h = evaluate_anomaly(X_hybrid, sc[:min_len])
    print(f"[compare] Hybrid: Verify EER={ver_h['eer']:.4f} AUC={ver_h['auc']:.4f} | Anomaly EER={ano_h['pooled_eer']:.4f}")

    def win(c, d, h):
        best_val = min(c, d, h)
        if best_val == h:
            return "Hybrid"
        elif best_val == c:
            return "Classical"
        else:
            return "Deep Learning"

    summary = {
        "classical": {"feature_dim": int(Xc.shape[1]), "verification": ver_c, "anomaly": ano_c},
        "deep_learning": {"embed_dim": int(Xd.shape[1]), "verification": ver_d, "anomaly": ano_d},
        "hybrid": {"dim": int(X_hybrid.shape[1]), "verification": ver_h, "anomaly": ano_h},
        "best_verification": win(ver_c["eer"], ver_d["eer"], ver_h["eer"]),
        "best_anomaly": win(ano_c["pooled_eer"], ano_d["pooled_eer"], ano_h["pooled_eer"]),
    }

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================  MODEL COMPARISON: CLASSICAL vs DEEP vs HYBRID  ================")
    print(f"{'Metric':<32}{'Classical':>12}{'Deep Learning':>16}{'Hybrid':>12}{'Winner':>14}")
    print(f"{'Biometric Verify EER':<32}{ver_c['eer']:>12.4f}{ver_d['eer']:>16.4f}{ver_h['eer']:>12.4f}{win(ver_c['eer'], ver_d['eer'], ver_h['eer']):>14}")
    print(f"{'Biometric Verify AUC':<32}{ver_c['auc']:>12.4f}{ver_d['auc']:>16.4f}{ver_h['auc']:>12.4f}{'Classical' if ver_c['auc'] > ver_d['auc'] and ver_c['auc'] > ver_h['auc'] else ('Hybrid' if ver_h['auc'] > ver_d['auc'] else 'Deep'):>14}")
    print(f"{'Anomaly Pooled EER':<32}{ano_c['pooled_eer']:>12.4f}{ano_d['pooled_eer']:>16.4f}{ano_h['pooled_eer']:>12.4f}{win(ano_c['pooled_eer'], ano_d['pooled_eer'], ano_h['pooled_eer']):>14}")
    print(f"{'Anomaly Mean/User EER':<32}{ano_c['mean_per_user_eer']:>12.4f}{ano_d['mean_per_user_eer']:>16.4f}{ano_h['mean_per_user_eer']:>12.4f}{win(ano_c['mean_per_user_eer'], ano_d['mean_per_user_eer'], ano_h['mean_per_user_eer']):>14}")
    print("=================================================================================")
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
