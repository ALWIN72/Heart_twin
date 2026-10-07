# Digital Heart Twin Digital Heart Twin

A personalized cardiac anomaly monitoring system built on smartphone
seismocardiography (SCG). A self-supervised foundation model learns general
cardiac-vibration features from accelerometer (SCG) and gyroscope signals,
then a lightweight **personal twin** per user learns what *that user's own*
heartbeat normally looks like — and flags meaningful deviations from their
own baseline, not from a population average.

## Pipeline overview

```
Raw recordings (Subject_*/Recording_*/scg.csv, Uncalibrated_scg.csv)
        |
        v
 Phase 1: Pretrain (src/training/pretrain.py)
   Masked-autoencoder pretraining of the shared foundation encoder
   (SCGEncoder) on ALL subjects, no labels required.
        |
        +----------------+----------------------+
        v                v                       v
 Phase 2: Denoiser   Phase 3: Harmonizer    Phase 3.5: Placement
 motion-artifact      device-invariant       classifier + spatial
 cancellation via     features via domain-   corrector
 SCG<->gyro cross-    adversarial training   (Sternum/Left/Right)
 attention            (DANN)
        |                |                       |
        +----------------+-----------------------+
                          v
              Phase 4: Personal Heart Twin
              (src/training/train_personal_twin.py)
              Per-user VAE head on top of the FROZEN foundation encoder.
                          |
                          v
              Phase 4.2: HealthMonitor
              (src/utils/anomaly_scorer.py)
              Production inference: placement -> harmonize ->
              personal anomaly score -> trend/alert
                          |
              +-----------+-----------+
              v                       v
   Phase 5: Benchmarking
   EER, AUC-ROC, lead time,
   false-alarm rate
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic dataset (dev/test fixture only -- see
#    "About the data" below) and run every phase end-to-end:
python run_pipeline.py --smoke_test

# 2. Run the test suite
python -m pytest tests/test_pipeline.py -v

# 3. Start the mobile recorder server (serves index.html + REST API)
python serve_recorder.py
# Then open https://<your-local-ip>:5001 on your phone (accept the
# self-signed certificate warning). HTTPS is required for device-motion
# sensor permission on iOS/Android.
```

## Using the real Digital Heart Twin dataset

Download the dataset and extract it so the directory layout looks like:

```
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
```

Then run the full pipeline against it:

```bash
python run_pipeline.py --data_path data/raw --epochs 200
```

Every phase can also be run independently -- see `src/training/*.py`, each
of which accepts `--data_path`, `--epochs`, `--batch_size`. Useful when you
want to retrain just one stage (e.g. only Phase 4 after collecting more
recordings for one user):

```bash
python -m src.training.pretrain --data_path data/raw --epochs 200
python -m src.training.finetune_denoiser --data_path data/raw --epochs 50
python -m src.training.finetune_harmonizer --data_path data/raw --epochs 50
python -m src.training.finetune_placement --data_path data/raw --epochs 30
python -m src.training.train_personal_twin --data_path data/raw --user_id 001
python -m src.evaluation.benchmark
```

## About the data: synthetic fixture vs. real dataset

This project ships a **synthetic data generator**
(`src/utils/synthetic_data.py`) used only to exercise and unit-test the
pipeline before real data is available. It writes a small dataset in the
exact directory layout the real loader expects, so every downstream
component (manifest building, windowing, training, evaluation)
runs identically on synthetic or real data -- only the *content* differs.
**Do not use results trained on synthetic data as a measure of real-world
model quality**; the synthetic signals are simplified pseudo-heartbeats,
not physiological recordings.

### On the "gyro" channel

The architecture is designed for 6 channels (3 SCG accelerometer axes + 3
gyroscope axes), but the current Digital Heart Twin release ships calibrated +
uncalibrated SCG only -- no real gyroscope stream. Rather than silently
zero-padding or skipping the gyro channels, `src/utils/preprocessing.py`
derives an explicit, clearly-labeled **gyro proxy** from the low-frequency
component that SCG calibration normally strips out (search the codebase
for `GYRO PROXY` to find the one place this happens). This keeps the
6-channel architecture exercised end-to-end today. The moment real
gyroscope data is released, swap it in by populating a `gyro.csv` per
recording -- `data_loader.py` already checks for and prefers a real `gyro`
column over the proxy.

### On placement labels

The dataset's `general_metadata.json` does not currently include a
Sternum/Left/Right placement label per recording. `finetune_placement.py`
detects this automatically and falls back to a self-supervised
synthetic-rotation pretext task (apply a known rotation, train the
classifier to recover which rotation was applied) instead of silently
training on fabricated labels. Once placement-labeled data exists (e.g. in
an optional `recording_metadata.json` with a `"placement"` field), the
script switches to fully supervised training automatically.

## Mobile recorder — `recorder/index.html` + `serve_recorder.py`

The smartphone capture layer is a **single-file progressive web app** (`recorder/index.html`)
served over HTTPS by `serve_recorder.py`. It runs entirely in the mobile browser — no app
install required.

### How to use it

1. Run `python serve_recorder.py` on your PC (generates a self-signed TLS cert on first run).
2. Open `https://<PC-IP>:5001` in Safari (iOS) or Chrome (Android) on your phone.
3. Accept the certificate warning (required for sensor permission).
4. Follow the **Guide tab** — it walks through every step.

### Tabs

| Tab | Purpose |
|---|---|
| **Record** | Enable motion sensors, use the signal coach to verify placement, then record SCG sessions |
| **Insights** | Live HRV analysis of the last recording — BPM, SDNN, RMSSD, pNN50, respiration, signal quality, deviation from personal baseline |
| **Breathe** | Guided 5.5 bpm slow-breathing session with live RSA (heart-rate variability swing) biofeedback |
| **History** | BPM trend chart across all sessions, per-recording anomaly badges, upload to PC / download as ZIP |
| **Dashboard** | Pulls `/dashboard_data` from the PC server and renders anomaly score trend, threshold line, and per-recording alerts |
| **Guide** | 8-step onboarding checklist — auto-detects completed steps from sensor state and server status |

### In-browser DSP (no server needed for local analysis)

All real-time signal processing runs in JavaScript inside the browser:

- **Resampling** — linear interpolation to a uniform 200 Hz grid (`resampleMag`)
- **High-pass filter** — moving-average baseline subtraction to remove gravity/drift (`movavgC`)
- **Beat detection** — envelope thresholding with refractory period (≥ 330 ms)
- **HRV metrics** — SDNN, RMSSD, pNN50 computed from RR intervals
- **Respiration rate** — autocorrelation periodicity on the low-frequency drift component
- **Poincaré plot** — SD1/SD2/LF-HF from RR scatter (shown in the Advanced HRV lab)
- **Signal quality score** — autocorrelation strength of the squared envelope, mapped to
  Poor / Fair / Good and displayed in the coach ring
- **Personal baseline deviation** — Z-score distance from a stored per-context baseline
  (BPM, SDNN, signal strength), persisted in `localStorage`

### REST API endpoints (served by `serve_recorder.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Saves `scg.csv`, `Uncalibrated_scg.csv`, `gyro.csv` to `data/raw/MSCardio/Subject_XXX/Recording_YY/` |
| `/twin_status` | GET | Returns training status and recording count for a subject |
| `/train_twin` | POST | Triggers `train_personal_twin.py` for a subject on the PC |
| `/twin_score` | POST | Runs `HealthMonitor` inference on an uploaded recording and returns KL anomaly score |
| `/dashboard_data` | GET | Returns all scored recordings with threshold and alert flags for the Dashboard tab |

### Data flow

```
Phone (browser)                         PC (serve_recorder.py)
  Sensors (250 Hz requested)    ──┐
  In-browser DSP (200 Hz)        │
  Record tab                     │  POST /upload  →  data/raw/MSCardio/
  History tab ───────────────────┘                     Subject_XXX/
  Guide tab   ──── GET /twin_status ─────────────  Recording_YY/
  Insights tab ─── POST /twin_score ─────────────    scg.csv
  Dashboard tab ── GET /dashboard_data ───────────   Uncalibrated_scg.csv
                   POST /train_twin ───────────────  gyro.csv
                                          ↓
                                  train_personal_twin.py
                                  anomaly_scorer.py (HealthMonitor)
```

### Signal coach

The **Record tab** includes a live signal-quality coach that runs while sensors are on
(even between recordings). It scores the current 8-second buffer and guides placement:

| Score | Ring colour | Meaning |
|---|---|---|
| ≥ 92 | Green | Good signal — safe to record |
| 58–91 | Amber | Almost there — adjust position |
| ≤ 57 | Red | Weak/noisy signal — reposition |

The score is the autocorrelation strength of the squared acceleration envelope,
normalized to 0–100.

---

## Project structure

```
recorder/
  index.html                    Single-file mobile web app (Record / Insights / Breathe /
                                  History / Dashboard / Guide tabs). All DSP in-browser JS.
serve_recorder.py               HTTPS server: serves recorder/, REST API for upload /
                                  train_twin / twin_score / dashboard_data endpoints.
Launch_DigitalHeartTwin.bat     One-click Windows launcher for serve_recorder.py
src/
  config.py                    Central hyperparameters & paths
  data_loader.py                Manifest building, subject-wise splits
  dataset.py                    PyTorch Dataset: CSV -> windowed 6-channel tensors
  models/
    encoder.py                   SCGEncoder: shared foundation backbone
    fusion_transformer.py        Cross-modal SCG<->gyro attention
    mae.py                        Masked autoencoder (Phase 1 pretraining objective)
    denoiser.py                   MotionArtifactCanceller (Phase 2)
    dann.py                        DeviceHarmonizer w/ gradient reversal (Phase 3)
    placement_classifier.py       Sternum/Left/Right classifier (Phase 3.5)
    spatial_transformer.py        Channel-mixing placement corrector (Phase 3.5)
    personal_twin.py              PersonalHeartTwin VAE (Phase 4)
  training/
    pretrain.py                   Phase 1 training loop
    finetune_denoiser.py          Phase 2 training loop
    finetune_harmonizer.py        Phase 3 training loop
    finetune_placement.py         Phase 3.5 training loop
    train_personal_twin.py        Phase 4 per-user training loop
  evaluation/
    benchmark.py                   Phase 5: EER, AUC-ROC, lead time, false-alarm rate
  utils/
    preprocessing.py               Resampling, filtering, windowing, gyro proxy
    synthetic_data.py              Dev/test-only synthetic dataset generator
    simulate_disease.py            Synthetic deterioration simulator (for eval)
    anomaly_scorer.py              HealthMonitor: production inference pipeline
tests/
  test_pipeline.py                End-to-end smoke tests for every phase
run_pipeline.py                   Orchestrates all phases in sequence
requirements.txt
```

## Design notes / deviations from the original roadmap sketch

A few places in the original architecture sketch didn't translate directly
into working code, and were redesigned rather than patched over:

- **Spatial correction** originally used `torch.affine_grid`/`grid_sample`,
  which are 2D/3D *image* resampling ops with no meaningful interpretation
  on a 1D multi-channel time series. Replaced with a physically motivated
  per-placement 3x3 channel-mixing (rotation) matrix applied at every
  timestep -- see `src/models/spatial_transformer.py` for the full
  rationale.
- **PersonalHeartTwin baseline storage** no longer silently recomputes and
  overwrites a "baseline" on every forward call. `set_baseline(...)` is now
  an explicit method that aggregates a batch of known-healthy recordings,
  and `anomaly_score(...)` raises clearly if called before a baseline has
  been set.
- **Placement classifier** used a hard-coded flatten size that only worked
  for one specific input length. Replaced with adaptive pooling so it works
  for any window length.
- **Layer freezing** in the denoiser ("freeze the first N parameters")
  didn't correspond to any meaningful architectural boundary. Replaced with
  freezing whole early transformer blocks.

## Metrics targets (Phase 5)

| Metric | Target | Meaning |
|---|---|---|
| EER | < 0.05 | How well a user's twin distinguishes their own recordings from others' |
| AUC-ROC | > 0.95 | Detection quality on simulated deterioration |
| Lead Time | > 14 days | How early the system flags simulated symptom onset |
| False Alarm Rate | < 0.02 | Fraction of healthy days incorrectly flagged |

Numbers from a smoke-test run on synthetic data are **not** meaningful
measures of real-world performance -- they only confirm the pipeline
executes and the metrics are computed correctly. Re-run
`python -m src.evaluation.benchmark` after training on real data with a
realistic number of epochs.
