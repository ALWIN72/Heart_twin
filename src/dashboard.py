"""
src/dashboard.py

Phase 6: Interactive Streamlit Dashboard.

Lets a user pick a subject, browse their recordings, run the full
Phase 1-4 pipeline (placement correction -> harmonization -> personal
twin anomaly scoring) on each recording in chronological order, and watch
the trend/alert evolve -- plus a "what if" simulated-deterioration view
built on the same disease simulator used in Phase 5 evaluation.

Run with:
    streamlit run src/dashboard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when streamlit runs this file directly (streamlit
# executes the script standalone, not as part of the `src` package).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.config import CHECKPOINT_DIR, SAMPLING_RATE_HZ, TwinConfig
from src.data_loader import build_multimodal_manifest
from src.dataset import SCGWindowDataset
from src.utils.anomaly_scorer import HealthMonitor
from src.utils.simulate_disease import simulate_heart_deterioration

st.set_page_config(page_title="MSCardio Digital Heart Twin", layout="wide")


@st.cache_data(show_spinner=False)
def get_manifest(data_path: str) -> pd.DataFrame:
    return build_multimodal_manifest(raw_data_path=data_path)


@st.cache_resource(show_spinner=False)
def get_monitor(user_id: str) -> HealthMonitor:
    return HealthMonitor.for_user(user_id)


def plot_waveform(signal: np.ndarray, title: str) -> go.Figure:
    """signal: (6, T) -> plot SCG (top) and gyro/gyro-proxy (bottom) channels."""
    t = np.arange(signal.shape[-1]) / SAMPLING_RATE_HZ
    fig = go.Figure()
    labels = ["SCG-x", "SCG-y", "SCG-z", "Gyro-x", "Gyro-y", "Gyro-z"]
    colors = ["#e74c3c", "#27ae60", "#2980b9", "#f39c12", "#8e44ad", "#16a085"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        visible = True if i < 3 else "legendonly"
        fig.add_trace(go.Scatter(x=t, y=signal[i], name=label, line=dict(color=color, width=1), visible=visible))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Normalized amplitude",
                       height=350, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    return fig


def plot_history(scores: list, threshold: float) -> go.Figure:
    fig = go.Figure()
    x = list(range(1, len(scores) + 1))
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines+markers", name="Anomaly score (KL)",
                              line=dict(color="#2980b9")))
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Alert threshold ({threshold})")
    fig.update_layout(title="Personal anomaly score over recordings", xaxis_title="Recording #",
                       yaxis_title="KL divergence from baseline", height=350,
                       margin=dict(l=10, r=10, t=40, b=10))
    return fig


def main():
    st.title("MSCardio Digital Heart Twin")
    st.caption("Personalized cardiac anomaly monitoring from smartphone seismocardiography (SCG)")

    with st.sidebar:
        st.header("Configuration")
        data_path = st.text_input("Raw data path", value="data/raw")
        manifest = get_manifest(data_path)

        if manifest.empty:
            st.error(f"No recordings found under `{data_path}`. Generate the synthetic dev "
                     f"dataset or point this at the real MSCardio dataset, then rerun.")
            st.stop()

        subjects = sorted(manifest["subject_id"].astype(str).unique())
        twin_available = [s for s in subjects if (CHECKPOINT_DIR / f"twin_user_{s}.pt").exists()]
        if not twin_available:
            st.warning("No subjects have a trained Personal Heart Twin yet. "
                       "Run `python -m src.training.train_personal_twin` first.")
            st.stop()

        user_id = st.selectbox("Subject", twin_available)
        st.caption(f"{len(twin_available)}/{len(subjects)} subjects have a trained twin.")

        mode = st.radio("View", ["Recording history", "Simulated deterioration"])

    monitor = get_monitor(user_id)
    twin_cfg = TwinConfig()

    sub_df = manifest[manifest["subject_id"].astype(str) == user_id].sort_values("recording_id")
    ds = SCGWindowDataset(sub_df)

    if mode == "Recording history":
        st.subheader(f"Subject {user_id} — recording-by-recording analysis")
        if len(ds) == 0:
            st.warning("No windows available for this subject.")
            st.stop()

        if st.button("Run full pipeline on all recordings", type="primary"):
            monitor.reset_history()
            progress = st.progress(0.0)
            results = []
            for i in range(len(ds)):
                x, meta = ds[i]
                res = monitor.analyze_recording(x)
                res["recording"] = i + 1
                results.append(res)
                progress.progress((i + 1) / len(ds))

            st.session_state[f"results_{user_id}"] = results

        results = st.session_state.get(f"results_{user_id}")
        if results:
            col1, col2 = st.columns([2, 1])
            with col1:
                scores = [r["score"] for r in results]
                st.plotly_chart(plot_history(scores, twin_cfg.anomaly_threshold), use_container_width=True)
            with col2:
                last = results[-1]
                st.metric("Latest anomaly score", f"{last['score']:.4f}")
                if "alert" in last:
                    if "WARNING" in last["alert"]:
                        st.error(last["alert"])
                    else:
                        st.success(last["alert"])
                else:
                    st.info(last.get("status", ""))

            idx = st.slider("Inspect recording #", 1, len(ds), len(ds))
            x_view, meta_view = ds[idx - 1]
            st.plotly_chart(plot_waveform(x_view.numpy(), f"Recording {idx} — {meta_view['platform']}"),
                             use_container_width=True)
        else:
            st.info("Click the button above to run the full Phase 1-4 pipeline on every recording.")

    else:
        st.subheader(f"Subject {user_id} — simulated deterioration stress test")
        st.caption("Synthetic day-by-day drift away from this subject's healthy baseline, used to "
                   "test whether the pipeline would catch a slow deterioration early. Not a "
                   "clinical model of any specific condition.")

        days = st.slider("Days to simulate", 10, 60, 30)
        onset_day = st.slider("Simulated symptom onset day", 1, days - 1, min(20, days - 1))

        if st.button("Run deterioration simulation", type="primary"):
            x0, _ = ds[0]
            healthy = x0.numpy()
            monitor.reset_history()
            scores = []
            waveforms = []
            for day, sig in enumerate(simulate_heart_deterioration(healthy, days=days, onset_day=onset_day)):
                x_t = torch.from_numpy(sig).float()
                res = monitor.analyze_recording(x_t)
                scores.append(res["score"])
                waveforms.append(sig)
            st.session_state[f"sim_{user_id}"] = {"scores": scores, "waveforms": waveforms, "onset_day": onset_day}

        sim = st.session_state.get(f"sim_{user_id}")
        if sim:
            fig = plot_history(sim["scores"], twin_cfg.anomaly_threshold)
            fig.add_vline(x=sim["onset_day"] + 1, line_dash="dot", line_color="orange",
                          annotation_text="Simulated symptom onset")
            st.plotly_chart(fig, use_container_width=True)

            flagged = [i for i, s in enumerate(sim["scores"]) if s > twin_cfg.anomaly_threshold]
            if flagged:
                first_flag = flagged[0]
                lead = sim["onset_day"] - first_flag
                if lead > 0:
                    st.success(f"First alert on day {first_flag + 1} — {lead} days BEFORE simulated onset.")
                else:
                    st.warning(f"First alert on day {first_flag + 1} — {-lead} days AFTER simulated onset.")
            else:
                st.error("No alert was ever triggered during the simulation.")

            day_view = st.slider("Inspect simulated day", 1, len(sim["waveforms"]), 1)
            st.plotly_chart(plot_waveform(sim["waveforms"][day_view - 1], f"Simulated day {day_view}"),
                             use_container_width=True)
        else:
            st.info("Click the button above to run the simulation.")


if __name__ == "__main__":
    main()
