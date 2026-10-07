"""
src/data_loader.py

Builds a manifest of every (subject, recording) pair in the Digital Heart Twin
dataset, and produces leakage-safe subject-wise train/val/test splits.

Expected raw layout (matches the Zenodo release, DOI 10.5281/zenodo.15657893):

    data/raw/Digital Heart Twin/
        Subject_001/
            general_metadata.json
            Recording_01/
                scg.csv
                Uncalibrated_scg.csv
            Recording_02/
                ...
        Subject_002/
            ...

If your local copy nests things slightly differently (e.g. an extra
top-level folder from how Zenodo zips it), point `raw_data_path` at the
folder that directly contains the `Subject_*` directories -- the loader
also tolerates `Subject_*` living one level deeper via `_find_subjects_dir`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_MANIFESTS, DATA_RAW


def _find_subjects_dir(raw_data_path: Path) -> Path:
    """Locate the directory that directly contains Subject_* folders."""
    raw_data_path = Path(raw_data_path)
    candidates = [
        raw_data_path,
        raw_data_path / "MSCardio",
        raw_data_path / "Digital Heart Twin",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("Subject_*")):
            return c
    # search up to 2 levels deep
    for c in raw_data_path.glob("*"):
        if c.is_dir() and any(c.glob("Subject_*")):
            return c
        if c.is_dir():
            for c2 in c.glob("*"):
                if c2.is_dir() and any(c2.glob("Subject_*")):
                    return c2
    raise FileNotFoundError(
        f"Could not find any 'Subject_*' folders under {raw_data_path}. "
        f"Expected e.g. {raw_data_path}/MSCardio/Subject_0001/... -- "
        f"check that the dataset was extracted correctly."
    )


def _safe_load_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _flatten_meta(meta: dict) -> dict:
    """
    Normalize subject metadata across releases. The real Digital Heart Twin dataset nests
    everything under a "general_info" object with capitalized keys
    (Sex/Race/Age/Height/Weight/device_name/platform); older/synthetic samples
    use flat lowercase keys (gender/smartphone_model/age/...). Return a single
    flat dict with canonical lowercase keys so downstream code is release-agnostic.
    """
    src = meta.get("general_info", meta) if isinstance(meta, dict) else {}
    pick = lambda *keys: next((src[k] for k in keys if k in src and src[k] not in (None, "", "None")), None)
    return {
        "device": pick("smartphone_model", "device_name", "device") or "unknown",
        "platform": str(pick("platform") or "unknown").lower(),
        "gender": pick("gender", "Sex", "sex") or "unknown",
        "race": pick("race", "Race") or "unknown",
        "age": _num(pick("age", "Age"), 1, 120),
        "height": _num(pick("height", "Height"), 80, 230),     # cm; drops junk like 17018
        "weight": _num(pick("weight", "Weight"), 25, 350),     # kg/lb; drops junk like 13878
        "placement": pick("placement", "Placement"),
    }


def _num(v, lo: float, hi: float):
    """Coerce to float and null out implausible / corrupt entries (the dataset has data-entry typos)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if lo <= f <= hi else np.nan


def _find_signal(rec_folder: Path, *names: str):
    """Case-tolerant lookup for a signal file (real data uses lowercase 'uncalibrated_scg.csv')."""
    existing = {p.name.lower(): p for p in rec_folder.iterdir() if p.is_file()}
    for n in names:
        p = existing.get(n.lower())
        if p is not None:
            return p
    return None


def build_multimodal_manifest(raw_data_path: str | Path = DATA_RAW,
                               out_path: str | Path = DATA_MANIFESTS / "multimodal_manifest.csv") -> pd.DataFrame:
    """
    Walk the raw dataset and extract per-recording metadata + file paths.
    Writes and returns a DataFrame manifest with full provenance tracking.
    """
    subjects_path = _find_subjects_dir(Path(raw_data_path))
    rows = []

    for subject_folder in sorted(subjects_path.glob("Subject_*")):
        # Standardize subject_id: strip Subject_ prefix and any leading/trailing spaces
        subject_id = subject_folder.name.split("_", 1)[1].strip()
        meta = _flatten_meta(_safe_load_metadata(subject_folder / "general_metadata.json"))

        recording_folders = sorted(subject_folder.glob("Recording_*"))
        if not recording_folders:
            continue

        for rec_folder in recording_folders:
            rec_id = rec_folder.name.split("_", 1)[1].strip()

            cal_path = _find_signal(rec_folder, "scg.csv")
            uncal_path = _find_signal(rec_folder, "Uncalibrated_scg.csv", "uncalibrated_scg.csv")
            gyro_path = _find_signal(rec_folder, "gyro.csv")

            if cal_path is None:
                continue

            rec_meta = _safe_load_metadata(rec_folder / "recording_metadata.json")
            rec_info = rec_meta.get("recording_info", rec_meta) if isinstance(rec_meta, dict) else {}
            placement = rec_info.get("placement", meta.get("placement", None))
            
            # Real provenance: timestamp and date-based session clustering
            rec_time = rec_info.get("recording_time", "")
            rec_timezone = rec_info.get("recording_timezone", "")
            
            # Extract date for session_id (e.g., '2024-11-06' from '2024-11-06_18-52-21')
            session_id = "session_01"
            if rec_time and "_" in str(rec_time):
                session_id = str(rec_time).split("_")[0]
            elif rec_time and "T" in str(rec_time):
                session_id = str(rec_time).split("T")[0]

            rows.append({
                "subject_id": subject_id,
                "recording_id": rec_id,
                "session_id": session_id,
                "recording_time": rec_time,
                "recording_timezone": rec_timezone,
                "cal_scg": str(cal_path),
                "uncal_scg": str(uncal_path) if uncal_path is not None else None,
                "gyro": str(gyro_path) if gyro_path is not None else None,
                "device": meta["device"],
                "platform": meta["platform"],
                "gender": meta["gender"],
                "age": meta["age"],
                "height": meta["height"],
                "weight": meta["weight"],
                "placement": placement,
            })

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def create_splits(manifest_path: str | Path = DATA_MANIFESTS / "multimodal_manifest.csv",
                   test_size: float = 0.3, val_fraction_of_temp: float = 0.5,
                   random_state: int = 42):
    """
    Strict subject-wise train/val/test split.
    Guarantees that subjects in train, val, and test are strictly disjoint.
    """
    if isinstance(manifest_path, pd.DataFrame):
        df = manifest_path.copy()
    else:
        df = pd.read_csv(manifest_path)
        
    if df.empty:
        raise ValueError(
            f"Manifest at {manifest_path} is empty. Run build_multimodal_manifest "
            f"first, or check that data/raw contains the Digital Heart Twin dataset."
        )

    subjects = np.unique(df["subject_id"].astype(str))
    if len(subjects) < 3:
        print(f"[create_splits] Only {len(subjects)} subject(s) found; "
              f"returning all data as train, empty val/test.")
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    train_subj, temp_subj = train_test_split(subjects, test_size=test_size, random_state=random_state)
    if len(temp_subj) < 2:
        val_subj, test_subj = temp_subj, np.array([])
    else:
        val_subj, test_subj = train_test_split(temp_subj, test_size=val_fraction_of_temp, random_state=random_state)

    train_df = df[df["subject_id"].astype(str).isin(train_subj)].reset_index(drop=True)
    val_df = df[df["subject_id"].astype(str).isin(val_subj)].reset_index(drop=True)
    test_df = df[df["subject_id"].astype(str).isin(test_subj)].reset_index(drop=True)

    # Verification of zero subject leakage
    set_train = set(train_df["subject_id"].astype(str))
    set_val = set(val_df["subject_id"].astype(str))
    set_test = set(test_df["subject_id"].astype(str))
    assert len(set_train & set_val) == 0, "FATAL: Subject leakage between train and val splits!"
    assert len(set_train & set_test) == 0, "FATAL: Subject leakage between train and test splits!"
    assert len(set_val & set_test) == 0, "FATAL: Subject leakage between val and test splits!"

    return train_df, val_df, test_df


def create_session_split(subject_df: pd.DataFrame, min_enrollment: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within-subject temporal/session split:
    Earlier sessions/recordings are allocated to ENROLLMENT.
    Later sessions/recordings are allocated to TEST.
    Never leaks future recordings into enrollment baseline.
    """
    df = subject_df.copy()
    if "recording_time" in df.columns and df["recording_time"].dropna().any():
        df = df.sort_values(["recording_time", "recording_id"]).reset_index(drop=True)
    else:
        df = df.sort_values("recording_id").reset_index(drop=True)

    sessions = df["session_id"].unique()
    if len(sessions) >= 2:
        # Session-disjoint split: first session(s) -> enroll, subsequent session(s) -> test
        enroll_sessions = sessions[: max(1, len(sessions) // 2)]
        enroll_df = df[df["session_id"].isin(enroll_sessions)].reset_index(drop=True)
        test_df = df[~df["session_id"].isin(enroll_sessions)].reset_index(drop=True)
    else:
        # Fall back to recording-index temporal split
        n_enroll = min_enrollment if len(df) > min_enrollment else max(1, len(df) // 2)
        enroll_df = df.iloc[:n_enroll].reset_index(drop=True)
        test_df = df.iloc[n_enroll:].reset_index(drop=True)

    return enroll_df, test_df


if __name__ == "__main__":
    manifest = build_multimodal_manifest()
    print(f"Built manifest with {len(manifest)} recordings across "
          f"{manifest['subject_id'].nunique() if not manifest.empty else 0} subjects.")
    train_df, val_df, test_df = create_splits(manifest)
    print(f"Train: {len(train_df)} recordings | Val: {len(val_df)} | Test: {len(test_df)}")

