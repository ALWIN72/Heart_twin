"""
src/evaluation/benchmark.py

Phase 5: Evaluation & Benchmarking.

Implements the four key metrics from the roadmap:
  - EER (Equal Error Rate): verification accuracy of the personal twin as
    a "is this really this user's heart signature" classifier (genuine vs
    impostor windows).
  - AUC-ROC: anomaly detection quality (distinguishing simulated
    deteriorated recordings from healthy ones).
  - Lead Time: days before a simulated symptom onset that the system first
    raises an alert.
  - False Alarm Rate: fraction of genuinely healthy days incorrectly flagged.

All four metrics depend on having trained PersonalHeartTwin checkpoints
(Phase 4) for the relevant subjects; run train_personal_twin.py first.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.config import CHECKPOINT_DIR, TwinConfig, get_device
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset, collate_scg
from src.utils.anomaly_scorer import load_model_bundle, load_user_twin
from src.utils.simulate_disease import simulate_heart_deterioration


# ---------------------------------------------------------------------------
# EER: verification accuracy
# ---------------------------------------------------------------------------
def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> tuple[float, float]:
    """
    genuine_scores: anomaly scores of a user's OWN held-out recordings
        against their OWN twin (should be low).
    impostor_scores: anomaly scores of OTHER users' recordings against
        this user's twin (should be high).

    Returns (eer, threshold_at_eer). EER is the point where False Accept
    Rate (impostor mistakenly accepted as genuine) equals False Reject
    Rate (genuine mistakenly rejected).
    """
    # Label convention for roc_curve: y=1 means "genuine" (low score should
    # predict this), so we use NEGATIVE score as the decision function.
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])
    y_score = -np.concatenate([genuine_scores, impostor_scores])  # higher = more "genuine-like"

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # EER is where fpr and fnr cross
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    eer_threshold = float(-thresholds[idx])
    return eer, eer_threshold


def evaluate_verification(test_subjects: list[str], manifest_df, checkpoint_dir: Path = CHECKPOINT_DIR,
                           max_windows_per_subject: int = 20) -> dict:
    """Computes EER by treating each subject's twin as a binary verifier against every other test subject."""
    device = get_device()
    bundle = load_model_bundle(checkpoint_dir, device)
    foundation = bundle["foundation"]

    subject_windows = {}
    for sid in test_subjects:
        sub_df = manifest_df[manifest_df["subject_id"].astype(str) == str(sid)]
        ds = SCGWindowDataset(sub_df)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=min(max_windows_per_subject, len(ds)), shuffle=False, collate_fn=collate_scg)
        x, _ = next(iter(loader))
        subject_windows[str(sid)] = x.to(device)

    eers = {}
    for sid in subject_windows:
        twin = load_user_twin(sid, foundation, checkpoint_dir, device)
        if twin is None:
            continue

        genuine_scores = twin.anomaly_score(subject_windows[sid]).cpu().numpy()

        impostor_scores = []
        for other_sid, other_x in subject_windows.items():
            if other_sid == sid:
                continue
            impostor_scores.append(twin.anomaly_score(other_x).cpu().numpy())
        if not impostor_scores:
            continue
        impostor_scores = np.concatenate(impostor_scores)

        eer, thresh = compute_eer(genuine_scores, impostor_scores)
        eers[sid] = {"eer": eer, "threshold": thresh,
                      "n_genuine": len(genuine_scores), "n_impostor": len(impostor_scores)}

    mean_eer = float(np.mean([v["eer"] for v in eers.values()])) if eers else float("nan")
    return {"per_subject": eers, "mean_eer": mean_eer}


# ---------------------------------------------------------------------------
# AUC-ROC + Lead Time + False Alarm Rate: simulated deterioration
# ---------------------------------------------------------------------------
def evaluate_early_warning(test_subjects: list[str], manifest_df, checkpoint_dir: Path = CHECKPOINT_DIR,
                            deterioration_days: int = 30, symptom_onset_day: int = 20) -> dict:
    """
    For each test subject with a trained twin: take one of their healthy
    windows, run it through `simulate_heart_deterioration` to generate a
    day-by-day degrading sequence, score every day with their personal
    twin, and report:
      - AUC-ROC distinguishing "past symptom onset" days from "before onset" days
      - lead time: first day the anomaly score crosses the alert threshold,
        relative to symptom_onset_day (negative = caught early)
      - false alarm rate: fraction of pre-onset (healthy) days incorrectly flagged
    """
    device = get_device()
    bundle = load_model_bundle(checkpoint_dir, device)
    foundation = bundle["foundation"]
    threshold = TwinConfig().anomaly_threshold

    results = {}
    for sid in test_subjects:
        twin = load_user_twin(str(sid), foundation, checkpoint_dir, device)
        if twin is None or not bool(twin.has_baseline.item()):
            continue

        sub_df = manifest_df[manifest_df["subject_id"].astype(str) == str(sid)]
        ds = SCGWindowDataset(sub_df)
        if len(ds) == 0:
            continue
        x0, _ = ds[0]
        healthy_signal = x0.numpy()  # (6, T)

        daily_scores = []
        for day, degraded in enumerate(simulate_heart_deterioration(healthy_signal, days=deterioration_days)):
            x_t = torch.from_numpy(degraded).float().unsqueeze(0).to(device)
            score = float(twin.anomaly_score(x_t).item())
            daily_scores.append(score)

        daily_scores = np.array(daily_scores)
        days = np.arange(len(daily_scores))
        y_true = (days >= symptom_onset_day).astype(int)

        try:
            auc = float(roc_auc_score(y_true, daily_scores)) if len(np.unique(y_true)) > 1 else float("nan")
        except ValueError:
            auc = float("nan")

        flagged_days = np.where(daily_scores > threshold)[0]
        first_flag_day = int(flagged_days[0]) if len(flagged_days) > 0 else None
        lead_time = (symptom_onset_day - first_flag_day) if first_flag_day is not None else None

        pre_onset_mask = days < symptom_onset_day
        false_alarms = int(np.sum(daily_scores[pre_onset_mask] > threshold))
        false_alarm_rate = false_alarms / max(1, pre_onset_mask.sum())

        results[str(sid)] = {
            "auc_roc": auc,
            "first_flag_day": first_flag_day,
            "lead_time_days": lead_time,
            "false_alarm_rate": float(false_alarm_rate),
            "daily_scores": daily_scores.tolist(),
        }

    valid_aucs = [v["auc_roc"] for v in results.values() if not np.isnan(v["auc_roc"])]
    valid_leads = [v["lead_time_days"] for v in results.values() if v["lead_time_days"] is not None]
    valid_far = [v["false_alarm_rate"] for v in results.values()]

    summary = {
        "mean_auc_roc": float(np.mean(valid_aucs)) if valid_aucs else float("nan"),
        "mean_lead_time_days": float(np.mean(valid_leads)) if valid_leads else float("nan"),
        "mean_false_alarm_rate": float(np.mean(valid_far)) if valid_far else float("nan"),
        "per_subject": results,
    }
    return summary


def run_full_benchmark(data_path: str = "data/raw", out_path: str = "checkpoints/benchmark_results.json"):
    manifest = build_multimodal_manifest(raw_data_path=data_path)
    all_subjects = sorted(manifest["subject_id"].astype(str).unique())

    # use subjects that actually have a trained twin checkpoint
    eligible = [s for s in all_subjects if (CHECKPOINT_DIR / f"twin_user_{s}.pt").exists()]
    print(f"[benchmark] {len(eligible)}/{len(all_subjects)} subjects have a trained personal twin.")

    if len(eligible) < 2:
        print("[benchmark] Need at least 2 subjects with trained twins for EER (impostor) testing. "
              "Run train_personal_twin.py for more users first.")
        eer_results = {"per_subject": {}, "mean_eer": float("nan")}
    else:
        eer_results = evaluate_verification(eligible, manifest)

    warning_results = evaluate_early_warning(eligible, manifest)

    summary = {
        "EER": eer_results["mean_eer"],
        "AUC_ROC": warning_results["mean_auc_roc"],
        "Lead_Time_days": warning_results["mean_lead_time_days"],
        "False_Alarm_Rate": warning_results["mean_false_alarm_rate"],
        "targets": {"EER": "< 0.05", "AUC_ROC": "> 0.95", "Lead_Time_days": "> 14", "False_Alarm_Rate": "< 0.02"},
        "details": {"verification": eer_results, "early_warning": warning_results},
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[benchmark] EER={summary['EER']:.4f} (target < 0.05)")
    print(f"[benchmark] AUC-ROC={summary['AUC_ROC']:.4f} (target > 0.95)")
    print(f"[benchmark] Lead Time={summary['Lead_Time_days']:.1f} days (target > 14)")
    print(f"[benchmark] False Alarm Rate={summary['False_Alarm_Rate']:.4f} (target < 0.02)")
    print(f"[benchmark] Full results saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_full_benchmark()
