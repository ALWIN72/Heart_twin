# Digital Heart Twin: Project Diary & Work Log Report

**Project Title:** Digital Heart Twin: Personalized Cardiac Anomaly Monitoring System from Smartphone Seismocardiography (SCG)  
**Student Name:** Alwin Varughese  
**Repository / Working Directory:** `C:\Users\alwin\Downloads\HEART_TWIN2`  
**Duration:** July 2026 – October 2026 (15 Weeks)  
**Academic / Review Cycle:** Final Year Capstone Project / Engineering Practicum  

---

## Executive Summary of Project Diary

The **Digital Heart Twin** project implements a personalized cardiovascular monitoring pipeline using smartphone seismocardiography (SCG). Rather than relying on population-averaged thresholds that yield high false alarms, the system combines a self-supervised transformer foundation encoder with a per-user Variational Autoencoder (VAE) "personal twin" that flags anomalies relative to an individual's own healthy baseline via closed-form KL divergence.

This project diary documents weekly milestones, technical activities, design decisions, mathematical implementations, software engineering tasks, testing regimens, and guide review checkpoints starting from the first week of July 2026 through mid-October 2026.

---

## Semester / Project Timeline Overview

| Week | Date Range | Project Phase | Core Focus Area | Status |
|:---:|:---|:---|:---|:---:|
| **Week 1** | Jul 01 – Jul 07, 2026 | Project Initiation & Literature Survey | Clinical SCG theory, physics of cardiac mechanics, problem formulation | Completed |
| **Week 2** | Jul 08 – Jul 14, 2026 | Dataset Analysis & Architecture Design | MSCardio dataset exploration, multi-axis IMU formats, system pipeline design | Completed |
| **Week 3** | Jul 15 – Jul 21, 2026 | Signal Processing Engine | Monotonic resampling, 0.8–30 Hz Butterworth bandpass filtering, windowing | Completed |
| **Week 4** | Jul 22 – Jul 28, 2026 | Signal Quality Index (SQI) & Denoising | Multi-band noise decomposition, kurtosis & template correlation gating | Completed |
| **Week 5** | Jul 29 – Aug 04, 2026 | Foundation Model Pretraining (Phase 1) | ViT-style Patch SCGEncoder, Masked Autoencoder (MAE) self-supervised loop | Completed |
| **Week 6** | Aug 05 – Aug 11, 2026 | Cross-Modal Denoising & Harmonization (Phase 2 & 3) | SCG-Gyro cross-attention denoiser, Domain-Adversarial (DANN) device invariance | Completed |
| **Week 7** | Aug 12 – Aug 18, 2026 | Spatial Placement & Coordinate Transformation (Phase 3.5) | Sternum/Left/Right placement classifier, spatial transformer channel-mixing | Completed |
| **Week 8** | Aug 19 – Aug 25, 2026 | Personal Heart Twin VAE (Phase 4) | Per-user latent space VAE, frozen encoder representations, baseline estimation | Completed |
| **Week 9** | Aug 26 – Sep 01, 2026 | Anomaly Scoring Engine & HealthMonitor (Phase 4.2) | Diagonal-Gaussian closed-form KL divergence, temporal alert thresholding | Completed |
| **Week 10** | Sep 02 – Sep 08, 2026 | Model Evaluation & Verification (Phase 5) | Equal Error Rate (EER), AUC-ROC, synthetic deterioration benchmarking | Completed |
| **Week 11** | Sep 09 – Sep 15, 2026 | Mobile Telemetry Web Application | HTTPS server, Generic Sensor API IMU capture (200 Hz), canvas oscilloscopes | Completed |
| **Week 12** | Sep 16 – Sep 22, 2026 | Biometric HeartPrint Engine | 20-piece Voronoi puzzle enrollment, progressive cardiac signature synthesis | Completed |
| **Week 13** | Sep 23 – Sep 29, 2026 | UI/UX Refinement & Guided Stepper | Layman-friendly 8-step journey, dual theme system (Tech Lime & Velvet Rose) | Completed |
| **Week 14** | Sep 30 – Oct 06, 2026 | Security, Authentication & Profiles | Persistent SHA-256 salted auth, session token management, profile modal | Completed |
| **Week 15** | Oct 07 – Oct 13, 2026 | Cloud Deployment & Offline-First Resilience | Vercel serverless integration, hybrid offline persistence, git sync & final audit | Completed |

---

## Detailed Weekly Diary Entries

### Week 1: July 01, 2026 – July 07, 2026
- **Phase:** Project Conception, Literature Review & Problem Formulation
- **Key Objectives:**
  - Investigate feasibility of smartphone accelerometer and gyroscope sensors for capturing micro-vibrations induced by myocardial contraction and aortic valve dynamics.
  - Review existing literature on Seismocardiography (SCG), Ballistocardiography (BCG), and Electrocardiography (ECG) correlations.
  - Formulate problem statement and justify the need for personalized thresholding over population-wide baselines.
- **Tasks & Methodologies:**
  - Studied mechanical cardiac phases: Mitral Valve Closure (MC), Isovolumic Contraction (IM), Aortic Valve Opening (AO), Rapid Ejection (RE), and Aortic Valve Closure (AC).
  - Identified core limitation in legacy telemedicine: inter-subject physiological variance causes high false-positive rates when using rigid universal cutoff thresholds.
  - Proposed the "Digital Heart Twin" paradigm: pretrain a foundation feature extractor across all subjects, then train a personal generative twin on healthy calibration data for each patient.
- **Deliverables:**
  - Literature survey summary document; project scope definition; mathematical formulation outline for anomaly detection via posterior-baseline divergence.
- **Challenges & Resolutions:**
  - *Challenge:* Smartphone sensors are prone to low signal-to-noise ratio (SNR) compared to clinical accelerometers.
  - *Resolution:* Decided on a multi-stage denoising architecture incorporating bandpass filtering, Signal Quality Index (SQI) gating, and cross-axis attention.
- **Supervisor / Guide Verification:** Approved conceptual proposal and initial architecture blueprint.

---

### Week 2: July 08, 2026 – July 14, 2026
- **Phase:** Dataset Exploration & Pipeline Specification
- **Key Objectives:**
  - Analyze the MSCardio dataset structure and format requirements.
  - Establish directory schemas, data loading conventions, and configuration parameters.
- **Tasks & Methodologies:**
  - Inspected MSCardio recording schema: multi-session recordings per subject containing calibrated and uncalibrated linear acceleration axes (`scg.csv`, `Uncalibrated_scg.csv`).
  - Formulated `src/config.py`: defined universal sampling rate ($f_s = 250\text{ Hz}$), window length ($N = 2500$ samples / $10\text{ s}$), patch length ($P = 125$ samples / $0.5\text{ s}$), resulting in $T = 20$ non-overlapping patches per window across 6 sensor channels.
  - Built manifest generator in `src/data_loader.py` to index subjects, sessions, file locations, and session quality metadata without memory leakage.
- **Tools Used:** Python 3.11/3.13, NumPy, Pandas, Pathlib.
- **Deliverables:** `src/config.py`, `src/data_loader.py`.
- **Supervisor Verification:** Dataset partitioning verified (subject-independent splits for foundation training vs. subject-specific splits for twin training).

---

### Week 3: July 15, 2026 – July 21, 2026
- **Phase:** Signal Preprocessing & Temporal Alignment
- **Key Objectives:**
  - Build robust mathematical signal cleaning routines for non-uniform sampling and physiological frequency isolation.
- **Tasks & Methodologies:**
  - Implemented monotonic time-grid resampling in `src/utils/preprocessing.py` using linear interpolation to handle timestamp jitter common in mobile sensor threads.
  - Designed zero-phase 4th-order Butterworth bandpass filter with cutoff frequencies $0.8\text{ Hz}$ to $30.0\text{ Hz}$ to isolate cardiac mechanical vibrations while eliminating baseline wander and high-frequency noise.
  - Integrated rolling window extraction with $50\%$ overlap, accompanied by dynamic gyroscope proxy derivation for multi-channel compatibility.
- **Tools Used:** SciPy Signal, NumPy.
- **Deliverables:** `src/utils/preprocessing.py`, `src/dataset.py` (`SCGWindowDataset`).
- **Challenges & Resolutions:**
  - *Challenge:* Edge effects in IIR Butterworth filtering on truncated windows.
  - *Resolution:* Implemented forward-backward zero-phase filtering (`scipy.signal.filtfilt`) with reflection padding to prevent phase distortion.
- **Supervisor Verification:** Frequency response and windowing dimensions verified against physiological specs.

---

### Week 4: July 22, 2026 – July 28, 2026
- **Phase:** Signal Quality Index (SQI) & Denoising Framework
- **Key Objectives:**
  - Develop gating mechanism to reject motion artifacts (walking, talking, sensor repositioning) before feeding data into neural networks.
- **Tasks & Methodologies:**
  - Formulated statistical quality metrics:
    - Kurtosis-based metric ($kSQI$) to identify non-physiological impulsive spikes.
    - Spectral energy ratio ($bSQI$) measuring signal power within the diagnostic cardiac band ($1.0 - 20.0\text{ Hz}$) relative to total power.
    - Amplitude variance thresholds ensuring minimum contact pressure.
  - Implemented decision logic: windows falling below quality criteria are tagged as invalid and excluded from baseline training.
- **Deliverables:** Signal quality scoring modules in preprocessing pipelines; test suite verifying artifact rejection.
- **Supervisor Verification:** Evaluated against clean vs. corrupted test samples.

---

### Week 5: July 29, 2026 – August 04, 2026
- **Phase 1:** Self-Supervised Foundation Model Pretraining
- **Key Objectives:**
  - Construct transformer backbone (`SCGEncoder`) and implement self-supervised Masked Autoencoder (MAE) training loop.
- **Tasks & Methodologies:**
  - Implemented 1D convolutional patch projection embedding ($125 \to 256$ dimensions) with learned 1D positional encodings.
  - Constructed multi-layer Transformer Encoder with multi-head self-attention ($d_{\text{model}} = 256, n_{\text{head}} = 8, \text{layers} = 4, d_{\text{ff}} = 512$).
  - Developed MAE pretraining module (`src/models/mae.py`): applied random patch masking at $75\%$ masking ratio. Decoder reconstructs masked patches; loss calculated strictly over masked tokens using Mean Squared Error (MSE).
  - Wrote training script `src/training/pretrain.py`.
- **Tools Used:** PyTorch, PyTorch Lightning/nn.Module, AdamW optimizer.
- **Deliverables:** `src/models/encoder.py`, `src/models/mae.py`, `src/training/pretrain.py`.
- **Supervisor Verification:** Verified loss convergence and attention map plausibility over unlabelled training windows.

---

### Week 6: August 05, 2026 – August 11, 2026
- **Phases 2 & 3:** Cross-Modal Denoising & Device Harmonization
- **Key Objectives:**
  - Eliminate sensor hardware discrepancy between calibrated and uncalibrated mobile devices.
  - Cancel body motion interference using cross-channel attention.
- **Tasks & Methodologies:**
  - Implemented `Denoiser` (`src/models/denoiser.py`) utilizing multi-head cross-attention between linear SCG channels and rotational gyroscope channels to dynamically subtract motion artifacts.
  - Implemented Domain-Adversarial Neural Network (DANN) harmonizer (`src/models/dann.py`) featuring a Gradient Reversal Layer (GRL).
  - Tuned adversarial objective: feature extractor minimizes reconstruction loss while maximizing domain discriminator classification loss, forcing latent representations to be device-invariant.
  - Wrote `src/training/train_denoiser.py` and `src/training/train_harmonizer.py`.
- **Deliverables:** `src/models/denoiser.py`, `src/models/dann.py`, corresponding training loops.
- **Supervisor Verification:** Gradient reversal backpropagation mathematically verified.

---

### Week 7: August 12, 2026 – August 18, 2026
- **Phase 3.5:** Spatial Placement Classifier & Transformation
- **Key Objectives:**
  - Compensate for anatomical placement differences (e.g., sternum vs. left mid-clavicular line vs. right thorax).
- **Tasks & Methodologies:**
  - Constructed placement classifier (`src/models/placement_classifier.py`) predicting sensor location from raw spectral-spatial features.
  - Built spatial transformer module (`src/models/spatial_transformer.py`) applying learned affine coordinate rotations and channel-mixing matrices to project off-center placements back to the canonical sternal plane.
  - Implemented placement training script `src/training/train_placement.py`.
- **Deliverables:** `src/models/placement_classifier.py`, `src/models/spatial_transformer.py`.
- **Challenges & Resolutions:**
  - *Challenge:* Non-linear phase shifts when sensor is rotated $90^\circ$ on the chest.
  - *Resolution:* Regularized transformation matrix with orthogonality constraints to preserve signal energy.
- **Supervisor Verification:** Classification accuracy verified on labeled placement splits.

---

### Week 8: August 19, 2026 – August 25, 2026
- **Phase 4:** Personal Heart Twin Architecture (Per-User VAE)
- **Key Objectives:**
  - Construct lightweight per-user Variational Autoencoder (VAE) head that operates over frozen foundation encoder features.
- **Tasks & Methodologies:**
  - Built `PersonalHeartTwin` (`src/models/personal_twin.py`):
    - Froze all foundation encoder weights (`requires_grad = False`).
    - Added user-specific encoder MLP ($256 \to 32 \to 8$) generating mean $\boldsymbol{\mu}$ and log-variance $\log\boldsymbol{\sigma}^2$ in latent space $\mathbb{R}^8$.
    - Implemented reparameterization trick: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \boldsymbol{\sigma}, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
    - Added user decoder MLP reconstructing foundation tokens.
  - Implemented `set_baseline(healthy_windows)`: collects calibration recordings, computes baseline mean $\boldsymbol{\mu}_{\text{base}}$ and pooled baseline variance $\boldsymbol{\sigma}_{\text{base}}^2$ in linear space.
  - Wrote training loop `src/training/train_personal_twin.py`.
- **Deliverables:** `src/models/personal_twin.py`, `src/training/train_personal_twin.py`.
- **Supervisor Verification:** Verified that freezing foundation encoder prevents catastrophic forgetting while keeping personal twin lightweight (<50 KB parameter size).

---

### Week 9: August 26, 2026 – September 01, 2026
- **Phase 4.2:** Anomaly Scoring & HealthMonitor Engine
- **Key Objectives:**
  - Formulate mathematically rigorous anomaly scoring function and real-time clinical monitoring service.
- **Tasks & Methodologies:**
  - Derived closed-form Kullback-Leibler (KL) divergence between sample posterior distribution $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and stored personal baseline $\mathcal{N}(\boldsymbol{\mu}_{\text{base}}, \boldsymbol{\Sigma}_{\text{base}})$:
    $$D_{\text{KL}} = \frac{1}{2} \sum_{k=1}^{D} \left[ \frac{\sigma_k^2 + (\mu_k - \mu_{\text{base},k})^2}{\sigma_{\text{base},k}^2} - 1 - \ln\frac{\sigma_k^2}{\sigma_{\text{base},k}^2} \right]$$
  - Constructed `HealthMonitor` class (`src/utils/anomaly_scorer.py`):
    - Chains placement classification $\to$ spatial transformation $\to$ DANN harmonization $\to$ VAE projection $\to$ KL anomaly scoring.
    - Implemented moving-average anomaly score tracking and threshold gating ($\tau \approx 0.045$).
- **Deliverables:** `src/utils/anomaly_scorer.py`, unit test validation suite in `tests/test_pipeline.py`.
- **Supervisor Verification:** Closed-form KL calculation verified term-by-term against Gaussian distribution theory.

---

### Week 10: September 02, 2026 – September 08, 2026
- **Phase 5:** Pipeline Benchmarking & Performance Validation
- **Key Objectives:**
  - Evaluate system performance, discriminative power, and end-to-end execution.
- **Tasks & Methodologies:**
  - Created `src/evaluation/benchmark.py` calculating Equal Error Rate (EER), Area Under the ROC Curve (AUC-ROC), false alarm rate, and early-warning lead times.
  - Implemented cardiac pathology simulation in `src/utils/simulate_disease.py` (introducing progressive contractility loss, ectopic premature beats, and valvular murmurs) to evaluate detection sensitivity.
  - Built unified orchestration pipeline `run_pipeline.py` with CLI flags (`--smoke_test`, `--all_phases`).
  - Executed automated test suite: verified all 8 unit tests passed in `tests/test_pipeline.py`.
- **Deliverables:** `src/evaluation/benchmark.py`, `run_pipeline.py`, benchmark metric logs.
- **Supervisor Verification:** Benchmarking methodology and automated test execution reviewed and signed off.

---

### Week 11: September 09, 2026 – September 15, 2026
- **Mobile Telemetry:** Real-Time Browser Recorder & HTTPS Server
- **Key Objectives:**
  - Create mobile web app allowing users to record actual chest seismocardiography using phone sensors.
- **Tasks & Methodologies:**
  - Built `serve_recorder.py`: Python HTTPS server with auto-generated self-signed certificates (mandatory for mobile browser sensor permissions).
  - Built `recorder/index.html`: implemented HTML5 Generic Sensor API (`Accelerometer`, `LinearAccelerationSensor`, `Gyroscope`) sampling at up to $200\text{ Hz}$.
  - Built real-time canvas oscilloscopes for live 3-axis motion visualization, sample counters, timer countdowns, and sound-based heartbeat audio synthesizers (`playLubDub`).
  - Added REST endpoints: `/upload`, `/subject_recordings`, `/load_recording_data`, `/twin_status`, `/twin_score`.
- **Deliverables:** `serve_recorder.py`, `recorder/index.html`.
- **Supervisor Verification:** Live accelerometer capture tested on mobile browser over local Wi-Fi.

---

### Week 12: September 16, 2026 – September 22, 2026
- **Biometric Identity:** HeartPrint Progressive Enrollment Engine
- **Key Objectives:**
  - Develop an intuitive visual representation of the user's emerging cardiovascular baseline.
- **Tasks & Methodologies:**
  - Conceived the "HeartPrint" engine (`src/heartprint.py`): biometric signature modeled as a 20-piece Voronoi puzzle shaped as a human heart.
  - Designed progressive enrollment logic: each uploaded recording updates stability metrics for puzzle pieces; once piece stability surpasses $85\%$, the piece transitions from hollow outline to glowing gradient.
  - Created interactive SVG canvas and REST endpoints `/heartprint_status` and `/heartprint_enroll`.
- **Deliverables:** `src/heartprint.py`, HeartPrint SVG canvas component in `recorder/index.html`.
- **Supervisor Verification:** Progressive enrollment logic and biometric stability tracking validated.

---

### Week 13: September 23, 2026 – September 29, 2026
- **UI/UX Modernization:** Layman-Friendly Stepper & Dual Color Themes
- **Key Objectives:**
  - Simplify complex medical jargon for regular users and implement high-contrast modern themes.
- **Tasks & Methodologies:**
  - Refactored UI from a cluttered interface into a guided 8-Step Interactive Journey:
    1. Connect $\to$ 2. Position $\to$ 3. Record $\to$ 4. Results $\to$ 5. Signature $\to$ 6. Breathe $\to$ 7. Train AI $\to$ 8. History.
  - Implemented breathing pacer (6 breaths/min relaxation pacer with biofeedback circle).
  - Implemented two curated design systems:
    - **Dark Theme (Tech Dark & Electric Lime):** `#222629` (Canvas), `#474B4F` (Surface), `#6B6E70` (Border), `#86C232` (Electric Lime Accent), `#61892F` (Deep Forest).
    - **Light Theme (Velvet Rose & Alabaster):** `#E3E2DF` (Alabaster Background), `#E3AFBC` (Rose Surface), `#5D001E` (Velvet Burgundy Ink), `#9A1750` (Deep Ruby), `#EE4C7C` (Coral Rose Accent).
- **Deliverables:** Updated `recorder/index.html` with theme switcher and responsive stepper.
- **Supervisor Verification:** User accessibility and interface responsiveness approved.

---

### Week 14: September 30, 2026 – October 06, 2026
- **Security & Data:** Patient Authentication & Profile Storage
- **Key Objectives:**
  - Provide secure patient account registration, login, and multi-profile switching.
- **Tasks & Methodologies:**
  - Designed lightweight, thread-safe JSON authentication backend (`users_db.json`) in `serve_recorder.py`.
  - Implemented secure password hashing using SHA-256 with cryptographically random 16-character per-user salts.
  - Developed session management issuing 30-day cryptographically secure URL-safe tokens (`secrets.token_urlsafe(32)`).
  - Built frontend user profile chip, dual-tab sign-in/register modal, patient profile editor, and auto-hydration engine that loads baseline data upon authentication.
  - Added guest fallback allowing demo exploration without credential requirements.
- **Deliverables:** Endpoints `/auth/register`, `/auth/login`, `/auth/me`, `/auth/profiles`, `/auth/update_profile`, `/auth/logout`; modal UI in `recorder/index.html`.
- **Supervisor Verification:** Security hashing, token expiration, and profile hydration verified.

---

### Week 15: October 07, 2026 – October 13, 2026
- **Production Readiness:** Cloud Deployment, Cross-Origin CORS & Final Audit
- **Key Objectives:**
  - Deploy application to Vercel serverless cloud infrastructure while resolving CORS and routing challenges.
- **Tasks & Methodologies:**
  - Built Vercel serverless WSGI entrypoint `api/index.py` and configured `vercel.json` rewrite routing.
  - Diagnosed and resolved HTML fallback routing bug (`Unexpected token '<', "<!DOCTYPE ..."`): configured explicit route query mapping in `vercel.json` (`/auth/:path*` $\to$ `/api/index?route=/auth/:path*`).
  - Added `OPTIONS` preflight headers (`Access-Control-Allow-Origin: *`, `Access-Control-Allow-Methods: GET, POST, OPTIONS`).
  - Designed hybrid offline-first client persistence: if cloud serverless cold-starts return non-JSON, browser seamlessly persists patient profiles in `localStorage`, guaranteeing zero user disruption.
  - Pushed commits to GitHub (`https://github.com/ALWIN72/Heart_twin.git`).
  - Conducted complete codebase audit and created final project documentation.
- **Deliverables:** `api/index.py`, `vercel.json`, `PROJECT_REPORT.md`, `PROJECT_DIARY_REPORT.md`.
- **Supervisor Verification:** Final project demonstration and code repository signed off.

---

## Technical Stack & Library Summary

| Component | Technologies & Frameworks | Purpose |
|---|---|---|
| **Programming Language** | Python 3.11 / 3.13 | Core backend, machine learning, serverless APIs |
| **Deep Learning Framework** | PyTorch, TorchVision | Transformer backbone, VAE twin, MAE pretraining |
| **Scientific Computing** | NumPy, SciPy (Signal) | Digital filtering, FFT, Butterworth IIR, numerical interpolation |
| **Data Architecture** | Pandas, Pathlib, JSON | Manifest indexing, recording datasets, metadata tracking |
| **Web Server & REST API** | Python `ThreadingHTTPServer`, WSGI, SSL | Local HTTPS server with self-signed TLS certificates |
| **Serverless Cloud Runtime** | Vercel Serverless Functions (`api/index.py`) | Production cloud web hosting and REST API endpoints |
| **Frontend Technologies** | HTML5, CSS3 (Vanilla Glassmorphism), Modern JavaScript (ES6+) | Mobile telemetry recorder, dashboard, HeartPrint SVG |
| **Browser APIs** | Generic Sensor API (`LinearAccelerationSensor`, `Gyroscope`), Web Audio API | 200 Hz mobile IMU recording, real-time lub-dub cardiac audio |
| **Version Control & CI/CD** | Git, GitHub | Multi-branch source control, commit tracking |

---

## Project Outcomes & Achievements

1. **End-to-End Functionality:** Developed a working system bridging mobile hardware sensors with deep learning cardiac modeling.
2. **Personalized Mathematical Formulation:** Validated that individual-specific baseline divergence via closed-form KL divergence mitigates universal threshold false alarms.
3. **Hardware Agnostic:** Integrated spatial transformation and domain-adversarial networks to minimize device and placement variations.
4. **Dual Deployment Capability:** Fully operational both locally via native HTTPS (`serve_recorder.py` on port 8443) and in the cloud via Vercel serverless integration (`api/index.py`).
5. **Layman Accessibility:** Created an intuitive 8-step guided health check workflow with biofeedback breathing assistance and dynamic biometric HeartPrint visualizations.

---

**Student Signature:** ___________________________  
**Faculty Guide / Supervisor Signature:** ___________________________  
**Date of Submission:** October 13, 2026
