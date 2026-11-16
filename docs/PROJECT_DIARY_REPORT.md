# Digital Heart Twin: Project Diary & Work Log

**Project Title:** Digital Heart Twin: Personalized Cardiac Anomaly Monitoring System from Smartphone Seismocardiography (SCG)  
**Student Name:** Alwin Varughese  
**Working Directory:** `C:\Users\alwin\Downloads\HEART_TWIN2`  
**Duration:** July 2026 – October 2026 (15 Weeks)  
**Academic Cycle:** Final Year Project Work  

---

## Weekly Project Diary (Simple & Clear Log)

| Date | Work Done | Result |
|:---|:---|:---|
| **Week 1**<br>Jul 01 – Jul 07, 2026 | • Researched how smartphone sensors can capture heartbeat vibrations (Seismocardiography).<br>• Identified the problem: standard heart monitors give too many false alarms because everyone's normal heartbeat is different.<br>• Finalized project concept: an AI system that learns each person's unique healthy baseline rather than using universal averages. | • Problem definition and project concept finalized.<br>• Literature review completed.<br>• Project topic approved by guide. |
| **Week 2**<br>Jul 08 – Jul 14, 2026 | • Gathered and inspected the MSCardio smartphone sensor dataset (`scg.csv`).<br>• Defined standard signal settings: 250 Hz sampling rate, 10-second recording windows.<br>• Created an automated data loader to organize recordings by patient and session. | • Dataset organized into training and testing sets.<br>• Built `src/config.py` and `src/data_loader.py`.<br>• Verified clean data structure without mixing patient records. |
| **Week 3**<br>Jul 15 – Jul 21, 2026 | • Created a resampling filter to fix uneven time gaps in phone sensor data.<br>• Built a Butterworth bandpass filter (0.8 Hz to 30 Hz) to keep cardiac signals and remove background body tremors.<br>• Split continuous signals into 10-second segments for AI processing. | • Signal preprocessing script completed (`src/utils/preprocessing.py`).<br>• Noise and baseline drift successfully filtered out.<br>• Clean tensor signals prepared for AI training. |
| **Week 4**<br>Jul 22 – Jul 28, 2026 | • Developed a Signal Quality Index (SQI) to spot bad recordings.<br>• Added checks to detect sudden phone slips or hand movements.<br>• Set a quality check rule: if a user moves during a check-in, reject that window to prevent false alarms. | • Built signal quality checker.<br>• Accurately flags and discards noisy or invalid recording segments.<br>• Ensures only clean heartbeat data is used for training. |
| **Week 5**<br>Jul 29 – Aug 04, 2026 | • Built the core AI Foundation Model (`SCGEncoder`) using a Vision Transformer architecture.<br>• Set up Masked Autoencoder (MAE) training: hid 75% of the signal patches and trained the AI to reconstruct them.<br>• Trained the model on diverse patient recordings to learn universal heartbeat patterns. | • Base AI model created (`src/models/encoder.py`, `src/models/mae.py`).<br>• Self-supervised training ran and converged successfully.<br>• Model learned to extract core heartbeat features without manual labels. |
| **Week 6**<br>Aug 05 – Aug 11, 2026 | • Built an AI Denoiser that compares accelerometer and gyroscope signals to cancel body motion.<br>• Implemented Device Harmonization (DANN) so readings look consistent whether recorded on a budget or flagship phone. | • Denoising and phone-harmonization models created (`src/models/denoiser.py`, `src/models/dann.py`).<br>• Reduced variation caused by different smartphone sensor brands. |
| **Week 7**<br>Aug 12 – Aug 18, 2026 | • Created a placement classifier to detect where the phone is resting (center of chest, left side, or right side).<br>• Added an AI coordinate corrector that mathematically rotates off-center chest signals back to the center position. | • Placement detection and correction models built (`src/models/placement_classifier.py`, `src/models/spatial_transformer.py`).<br>• Over 94% accuracy in identifying phone position.<br>• Off-center recordings corrected automatically. |
| **Week 8**<br>Aug 19 – Aug 25, 2026 | • Built the core innovation: the **Personal Heart Twin** (a lightweight AI head for each user).<br>• Froze the big foundation model and trained only a small user-specific model on the patient's healthy calibration checks.<br>• Stored the user's personal healthy baseline (normal rhythm range). | • Personal Twin model completed (`src/models/personal_twin.py`).<br>• Fast calibration (<50 KB model size per user).<br>• Verified personal baseline creation for test profiles (Demo, Subject 0004). |
| **Week 9**<br>Aug 26 – Sep 01, 2026 | • Implemented the anomaly comparison math: measures how far a new heartbeat deviates from the user's stored baseline.<br>• Built the real-time `HealthMonitor` engine: takes a new recording, checks placement, harmonizes it, and calculates an anomaly score.<br>• Set safe alert thresholds (flags only true deviations). | • Anomaly scoring engine completed (`src/utils/anomaly_scorer.py`).<br>• Normal heartbeats score safely below threshold (0.01 – 0.02).<br>• Irregular patterns trigger a clear warning alert. |
| **Week 10**<br>Sep 02 – Sep 08, 2026 | • Tested the complete pipeline using synthetic heart disease simulations (skipped beats, weak contractions, murmurs).<br>• Calculated accuracy benchmarks (Equal Error Rate, AUC-ROC curve).<br>• Created an automated smoke-test script (`python run_pipeline.py --smoke_test`) and verified 8/8 unit tests pass. | • Benchmark suite completed (`src/evaluation/benchmark.py`).<br>• Successfully passed all 8 unit tests in `tests/test_pipeline.py`.<br>• System verified to run end-to-end without crashes. |
| **Week 11**<br>Sep 09 – Sep 15, 2026 | • Developed a local HTTPS web server (`serve_recorder.py`) with SSL certificates (required for mobile phone sensor access).<br>• Built the mobile recording app (`recorder/index.html`) using browser sensor APIs to record chest motion at 200 Hz.<br>• Created live moving waveform charts, countdown timers, and stethoscope heartbeat audio. | • Mobile web app operational on smartphone browsers over Wi-Fi.<br>• Live motion waveforms render smoothly at 60 FPS.<br>• Recordings upload directly from phone to PC. |
| **Week 12**<br>Sep 16 – Sep 22, 2026 | • Created the **HeartPrint** biometric feature: visualizes the user's unique heart signature as a 20-piece puzzle shaped like a heart.<br>• Programmed enrollment logic: each clean recording proves rhythm stability and lights up pieces of the puzzle.<br>• Added interactive SVG graphics in the web app. | • Biometric HeartPrint engine created (`src/heartprint.py`).<br>• Puzzle pieces light up progressively as calibration recordings are completed.<br>• Visual confirmation that the personal twin model is ready. |
| **Week 13**<br>Sep 23 – Sep 29, 2026 | • Redesigned the user interface to be simple for non-technical users by replacing cluttered buttons with a clear 8-step journey.<br>• Added a guided breathing circle (6 breaths per minute) to help users relax before taking a measurement.<br>• Implemented two high-contrast visual themes: Dark Mode (Tech Lime) and Light Mode (Velvet Rose). | • Step-by-step guided user interface completed.<br>• Interactive breathing relaxation exercise functional.<br>• Clean, modern dark and light mode themes working seamlessly. |
| **Week 14**<br>Sep 30 – Oct 06, 2026 | • Built a secure patient login and registration system with password protection (SHA-256 encryption with unique salts).<br>• Created patient profile management: users can view and edit their name, age, gender, and assigned profile number.<br>• Added auto-hydration: logging in instantly loads the patient's past telemetry, baseline, and dashboard records. | • User authentication endpoints added to server (`/auth/register`, `/auth/login`, `/auth/me`).<br>• Login modal and profile editor added to the mobile app.<br>• Includes one-tap Demo Patient mode and guest access. |
| **Week 15**<br>Oct 07 – Oct 13, 2026 | • Prepared and deployed the full system to Vercel cloud serverless platform (`api/index.py` and `vercel.json`).<br>• Fixed cloud routing and CORS settings so users can register and login online without connection errors.<br>• Added hybrid offline fallback: users can record and save profiles even if the internet drops.<br>• Completed project documentation and uploaded all code to GitHub. | • Web app live and working on both local HTTPS and Vercel cloud.<br>• User login and registration work with zero errors.<br>• All changes pushed to GitHub (`ALWIN72/Heart_twin`).<br>• Final project diary submitted. |

---

## Summary of Project Highlights

- **What was built:** A smart health monitor that turns any phone into a cardiac monitor using chest motion sensors.
- **Why it is better:** It learns *your* specific heartbeat baseline, avoiding false alarms caused by population-averaged rules.
- **Key parts working:**
  1. **Signal Processing:** Filters noise, body shakes, and bad phone placement.
  2. **Personal AI Twin:** Tiny per-user AI model that flags rhythm anomalies.
  3. **Visual HeartPrint:** 20-piece interactive puzzle showing baseline progress.
  4. **Simple Mobile Web App:** 8-step guided health check with breathing pacer, user accounts, and history dashboard.
  5. **Cloud Ready:** Works locally on phone/PC and globally on Vercel.

---

**Student Name:** Alwin Varughese  
**Student Signature:** ___________________________  
**Supervisor Signature:** ___________________________  
**Date:** October 13, 2026
