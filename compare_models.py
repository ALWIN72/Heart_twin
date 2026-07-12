"""
compare_models.py

Head-to-head on the REAL MSCardio data, classical-ML vs deep-learning, using the
*identical* evaluation protocol so the comparison is fair:

  - Classical features : 48 hand-engineered SCG/HRV features per recording.
  - DL embeddings      : the foundation SCGEncoder (MAE-pretrained on the real
                         data here) mean-pooled to a 256-d vector per recording.

Both feature sets are run through the SAME leave-one-recording-out biometric
verification and per-user anomaly evaluation. Whichever gives the lower EER wins.

Run:  python compare_models.py --data_path MSCardio --epochs 12 --max_steps 350
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from src.config import CHECKPOINT_DIR, ModelConfig, get_device
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset
from src.models.encoder import SCGEncoder
from src.training.pretrain import run_pretraining
from src.evaluation.classical_baseline import extract_features, evaluate_verification, evaluate_anomaly


def standardize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def classical_matrix(m):
    X, subj = [], []
    for i, (_, row) in enumerate(m.iterrows()):
        v = extract_features(row["cal_scg"])
        if v is not None:
            X.append(v); subj.append(str(row["subject_id"]))
        if (i + 1) % 150 == 0:
            print(f"  [classical] {i + 1}/{len(m)}")
    return standardize(np.vstack(X)), subj


def dl_matrix(m, encoder, device):
    encoder.eval()
    X, subj = [], []
    rows = list(m.iterrows())
    for i, (_, row) in enumerate(rows):
        df = m[(m["subject_id"] == row["subject_id"]) & (m["recording_id"] == row["recording_id"])]
        ds = SCGWindowDataset(df)
        if len(ds) == 0:
            continue
        xs = torch.stack([ds[j][0] for j in range(len(ds))], dim=0).to(device)
        with torch.no_grad():
            e = encoder.encode_pooled(xs).mean(dim=0).cpu().numpy()
        X.append(e); subj.append(str(row["subject_id"]))
        if (i + 1) % 150 == 0:
            print(f"  [dl-emb] {i + 1}/{len(rows)}")
    return standardize(np.vstack(X)), subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="MSCardio")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=350)
    ap.add_argument("--out_path", type=str, default="checkpoints/compare_results.json")
    args = ap.parse_args()
    device = get_device()
    print(f"[compare] device={device}")

    m = build_multimodal_manifest(raw_data_path=args.data_path)
    print(f"[compare] {len(m)} recordings, {m['subject_id'].nunique()} subjects")

    # ---- Classical ----
    print("[compare] extracting classical features…")
    Xc, sc = classical_matrix(m)
    ver_c = evaluate_verification(Xc, sc)
    ano_c = evaluate_anomaly(Xc, sc)
    print(f"[compare] classical: verify EER={ver_c['eer']:.4f} | anomaly EER={ano_c['pooled_eer']:.4f}")

    # ---- DL: pretrain foundation encoder on the real data, then embeddings ----
    print(f"[compare] pretraining foundation encoder on real data (epochs={args.epochs}, max_steps={args.max_steps})…")
    run_pretraining(data_path=args.data_path, epochs=args.epochs, batch_size=64,
                    checkpoint_name="encoder_real.pt", max_steps=args.max_steps)
    cfg = ModelConfig()
    enc = SCGEncoder(cfg)
    enc.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_real.pt", map_location="cpu", weights_only=True))
    enc.to(device)
    print("[compare] extracting DL embeddings…")
    Xd, sd = dl_matrix(m, enc, device)
    ver_d = evaluate_verification(Xd, sd)
    ano_d = evaluate_anomaly(Xd, sd)
    print(f"[compare] DL: verify EER={ver_d['eer']:.4f} | anomaly EER={ano_d['pooled_eer']:.4f}")

    summary = {
        "classical": {"feature_dim": int(Xc.shape[1]), "verification": ver_c, "anomaly": ano_c},
        "deep_learning": {"embed_dim": int(Xd.shape[1]), "verification": ver_d, "anomaly": ano_d},
    }
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(summary, f, indent=2)

    def win(c, d):
        return "classical" if c < d else ("DL" if d < c else "tie")
    print("\n================  CLASSICAL  vs  DEEP LEARNING  (real MSCardio)  ================")
    print(f"{'metric':<34}{'classical':>14}{'deep learning':>16}{'winner':>10}")
    print(f"{'Biometric verification  EER':<34}{ver_c['eer']:>14.4f}{ver_d['eer']:>16.4f}{win(ver_c['eer'],ver_d['eer']):>10}")
    print(f"{'Biometric verification  AUC':<34}{ver_c['auc']:>14.4f}{ver_d['auc']:>16.4f}{win(-ver_c['auc'],-ver_d['auc']):>10}")
    print(f"{'Personal anomaly  pooled EER':<34}{ano_c['pooled_eer']:>14.4f}{ano_d['pooled_eer']:>16.4f}{win(ano_c['pooled_eer'],ano_d['pooled_eer']):>10}")
    print(f"{'Personal anomaly  mean/user EER':<34}{ano_c['mean_per_user_eer']:>14.4f}{ano_d['mean_per_user_eer']:>16.4f}{win(ano_c['mean_per_user_eer'],ano_d['mean_per_user_eer']):>10}")
    print("================================================================================")
    print(f"Saved -> {args.out_path}")


if __name__ == "__main__":
    main()
