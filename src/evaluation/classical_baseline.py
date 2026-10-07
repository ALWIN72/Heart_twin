"""
src/evaluation/classical_baseline.py

A classical-ML baseline for the Digital Heart Twin dataset — the model the deep-learning
twin must beat. Pure feature-engineering + distance/one-class scoring, no
training of a big network. Two label-free-but-measurable tasks:

  1. Biometric verification  — "is this recording the same person?" Uses subject
     identity (which the dataset HAS). Leave-one-recording-out: a probe is scored
     against its own subject's template (genuine) and every other subject's
     template (impostor). Reported as EER / AUC.

  2. Personal anomaly twin   — "is this recording normal for ME?" For subjects
     with enough recordings, a per-user baseline (first N recordings) defines a
     diagonal-Mahalanobis normal model; a held-out own recording (genuine, should
     score low) is separated from other people's recordings (impostor, high).
     Reported as EER / AUC — directly comparable to the DL twin's EER.

Run:  python -m src.evaluation.classical_baseline --data_path Digital Heart Twin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal as sps, stats
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import CHECKPOINT_DIR
from src.data_loader import build_multimodal_manifest
from src.dataset import _load_scg_csv
from src.utils.preprocessing import bandpass_cardiac, resample_signal

FS = 100.0  # analysis rate (matches the real data's native ~100 Hz)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _bandpower(f, P, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapz(P[m], f[m])) if m.any() else 0.0


def _feat_1d(s: np.ndarray, fs: float = FS) -> list[float]:
    s = s - np.mean(s)
    out = [float(np.std(s)), float(np.sqrt(np.mean(s ** 2))), float(stats.iqr(s)),
           float(stats.skew(s)), float(stats.kurtosis(s)),
           float(np.mean((s[:-1] * s[1:]) < 0))]  # zero-crossing rate
    f, P = sps.welch(s, fs=fs, nperseg=int(min(len(s), 512)))
    tot = float(np.trapz(P, f)) + 1e-12
    for lo, hi in [(0.8, 4), (4, 12), (12, 30)]:
        out.append(_bandpower(f, P, lo, hi) / tot)
    out.append(float(np.sum(f * P) / (np.sum(P) + 1e-12)))           # spectral centroid
    band = (f >= 0.7) & (f <= 3.5)
    out.append(float(f[band][np.argmax(P[band])]) if band.any() and P[band].size else 0.0)  # HR-band peak
    return out


def _hrv(mag: np.ndarray, fs: float = FS) -> list[float]:
    """Heart rate + HRV from the SCG magnitude envelope."""
    s = mag - np.mean(mag)
    env = np.convolve(s ** 2, np.ones(max(2, int(fs * 0.08))) / max(2, int(fs * 0.08)), mode="same")
    # autocorr HR (40-180 bpm)
    e = env - env.mean()
    ac0 = float(np.dot(e, e)) + 1e-9
    lo, hi = int(0.33 * fs), int(1.5 * fs)
    best, bv = -1, -1.0
    for lag in range(lo, min(hi, len(e) - 1)):
        v = float(np.dot(e[:-lag], e[lag:])) / ac0
        if v > bv:
            bv, best = v, lag
    hr = 60.0 * fs / best if best > 0 else 0.0
    # beat peaks for HRV
    thr = env.mean() + 0.6 * env.std()
    refr = int(0.33 * fs)
    beats, last = [], -refr
    for i in range(1, len(env) - 1):
        if env[i] > thr and env[i] >= env[i - 1] and env[i] > env[i + 1] and i - last >= refr:
            beats.append(i); last = i
    rr = np.diff(beats) / fs * 1000.0 if len(beats) > 1 else np.array([])
    sdnn = float(np.std(rr)) if rr.size else 0.0
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size > 1 else 0.0
    return [hr, sdnn, rmssd, float(bv)]


def extract_features(cal_path: str) -> np.ndarray | None:
    try:
        arr, fs = _load_scg_csv(cal_path)               # (3, T), native fs
        arr = resample_signal(arr, fs, FS)              # -> uniform 100 Hz
        arr = bandpass_cardiac(arr, fs=FS, low_hz=0.8, high_hz=30.0)
        if arr.shape[-1] < FS * 3:
            return None
        mag = np.sqrt(np.sum(arr ** 2, axis=0))
        feats: list[float] = []
        for ax in range(3):
            feats += _feat_1d(arr[ax])
        feats += _feat_1d(mag)
        feats += _hrv(mag)
        v = np.array(feats, dtype=float)
        return v if np.all(np.isfinite(v)) else np.nan_to_num(v)
    except Exception as e:  # noqa: BLE001
        print(f"[classical] skip {cal_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def compute_eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """genuine: low scores (matches); impostor: high. Returns (eer, auc)."""
    y = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    s = -np.concatenate([genuine, impostor])            # higher = more genuine-like
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    auc = float(roc_auc_score(y, s))
    return eer, auc


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# Task 1: biometric verification
# ---------------------------------------------------------------------------
def evaluate_verification(X, subj):
    Xn = _l2(X)
    subj = np.asarray(subj)
    uniq = np.unique(subj)
    templates = {s: Xn[subj == s].mean(axis=0) for s in uniq}
    genuine, impostor = [], []
    for s in uniq:
        idx = np.where(subj == s)[0]
        if len(idx) < 2:
            continue
        for i in idx:
            tmpl = Xn[idx[idx != i]].mean(axis=0)             # leave-one-out own template
            genuine.append(1 - float(np.dot(Xn[i], tmpl) / (np.linalg.norm(tmpl) + 1e-9)))
            for s2 in uniq:
                if s2 == s:
                    continue
                t2 = templates[s2]
                impostor.append(1 - float(np.dot(Xn[i], t2) / (np.linalg.norm(t2) + 1e-9)))
    eer, auc = compute_eer(np.array(genuine), np.array(impostor))
    return {"eer": eer, "auc": auc, "n_genuine": len(genuine), "n_impostor": len(impostor),
            "n_subjects_tested": int(sum(np.sum(subj == s) >= 2 for s in uniq))}


# ---------------------------------------------------------------------------
# Task 2: personal anomaly twin (per-user diagonal Mahalanobis)
# ---------------------------------------------------------------------------
def evaluate_anomaly(X, subj, n_baseline=5, min_rec=6):
    subj = np.asarray(subj)
    mu_g = X.mean(axis=0)
    sd_g = X.std(axis=0) + 1e-6                            # population scale (robust diag Mahalanobis)
    Z = (X - mu_g) / sd_g
    uniq = np.unique(subj)
    genuine, impostor, eers = [], [], {}
    for s in uniq:
        idx = np.where(subj == s)[0]
        if len(idx) < min_rec:
            continue
        base = Z[idx[:n_baseline]]
        held = idx[n_baseline:]
        mu = base.mean(axis=0)
        score = lambda z: float(np.sqrt(np.mean((z - mu) ** 2)))   # distance from user's normal
        g = [score(Z[i]) for i in held]
        imp = [score(Z[i]) for i in np.where(subj != s)[0]]
        genuine += g; impostor += imp
        e, a = compute_eer(np.array(g), np.array(imp))
        eers[str(s)] = {"eer": e, "auc": a, "n_held": len(g)}
    eer, auc = compute_eer(np.array(genuine), np.array(impostor))
    mean_eer = float(np.nanmean([v["eer"] for v in eers.values()])) if eers else float("nan")
    return {"pooled_eer": eer, "pooled_auc": auc, "mean_per_user_eer": mean_eer,
            "n_eligible_subjects": len(eers), "per_user": eers}


def run(data_path="Digital Heart Twin", out_path="checkpoints/classical_results.json"):
    print("[classical] building manifest…")
    m = build_multimodal_manifest(raw_data_path=data_path)
    if m.empty:
        raise RuntimeError(f"No data at {data_path}")
    print(f"[classical] {len(m)} recordings, {m['subject_id'].nunique()} subjects. Extracting features…")
    X, subj = [], []
    for i, (_, row) in enumerate(m.iterrows()):
        v = extract_features(row["cal_scg"])
        if v is not None:
            X.append(v); subj.append(str(row["subject_id"]))
        if (i + 1) % 100 == 0:
            print(f"  …{i + 1}/{len(m)}")
    X = np.vstack(X)
    # standardize then verify / anomaly
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    print(f"[classical] feature matrix {X.shape}")
    ver = evaluate_verification(X, subj)
    ano = evaluate_anomaly(X, subj)
    summary = {"task_biometric_verification": ver, "task_personal_anomaly": ano,
               "feature_dim": int(X.shape[1]), "n_recordings": int(X.shape[0])}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n==== CLASSICAL BASELINE (real data) ====")
    print(f"Biometric verification : EER={ver['eer']:.4f}  AUC={ver['auc']:.4f}  "
          f"({ver['n_subjects_tested']} subjects, {ver['n_genuine']} genuine / {ver['n_impostor']} impostor)")
    print(f"Personal anomaly twin  : pooled EER={ano['pooled_eer']:.4f}  AUC={ano['pooled_auc']:.4f}  "
          f"mean per-user EER={ano['mean_per_user_eer']:.4f}  ({ano['n_eligible_subjects']} eligible subjects)")
    print(f"Saved -> {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Classical ML baseline for Digital Heart Twin")
    ap.add_argument("--data_path", type=str, default="Digital Heart Twin")
    ap.add_argument("--out_path", type=str, default="checkpoints/classical_results.json")
    args = ap.parse_args()
    run(args.data_path, args.out_path)
