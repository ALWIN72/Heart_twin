# Digital Heart Twin Digital Heart Twin — Advanced Architecture & Roadmap to "Fully Working"

*A research-grade blueprint for taking the current prototype to a state-of-the-art, real-world
system. Synthesized from six expert deep-dives (signal/data, foundation-model architecture,
personalization & anomaly theory, evaluation rigor, on-device/MLOps, clinical/regulatory).
Companion to `PROJECT_REPORT.md`, which audits the current code.*

---

## 0. The thesis

The current prototype is a competent **scaffold** wired to an **uncalibrated, leakage-validated,
semantics-free** scoring path. The route to "works in real life" is *not* a bigger anomaly score —
it is:

> a **quality-gated, beat-centric, context-conditioned** front end →
> a **transfer-pretrained, cycle-aware** encoder →
> a **per-user *calibrated density*** with conformal false-alarm control and change-point temporal logic →
> a **de-scoped, abstaining, on-device** product backed by **honest evidence.**

All six experts independently converged on the same three load-bearing fixes:

1. **Gate junk before it scores** (signal quality index → reject/abstain).
2. **Condition "normal" on context** (activity, posture, HR) so a workout isn't flagged as cardiac.
3. **Calibrate the decision per-user and de-leak the evaluation.**

Everything else amplifies these three.

---

## 1. Unified target architecture (end-to-end)

Read this as a single data contract flowing left → right. **Bold = change vs the current prototype.**

### Layer 1 — Signal front end (per recording)
- **Unit/rate calibration** — detect g vs m/s² from the gravity magnitude of the still segment;
  reject implausible inferred `fs` instead of silently defaulting to 250 Hz.
- **Monotonic-grid resampling** with a per-sample validity mask (phone IMU streams drop samples;
  the current median-Δt assumption is false).
- **Gravity/orientation split** (Madgwick/complementary filter) → linear acceleration **plus a free
  posture state** from the gravity vector.
- **Robust normalization** — replace global per-window z-norm with **median/MAD** scaling:
  `x_norm = (x − median) / (1.4826·MAD + ε)`. *(A 1 s motion spike currently rescales the whole
  10 s window and crushes the cardiac signal.)*
- **Multi-band decomposition** — cardiac SCG band; a **real respiration channel (0.1–0.5 Hz)** that
  *replaces* the mislabeled gyro proxy; a broadband motion band for quality scoring.
- **SQI hard gate** (highest-value single change) — fuse `kSQI` (kurtosis), `bSQI` (in-band energy
  fraction), `pSQI` (beat-template correlation), and a motion/ENMO score into `accept / flag /
  reject`. **Only `accept` reaches the encoder, twin, and benchmark.** This is the dominant
  real-world false-alarm source today and is entirely unhandled.
- **Beat (AO-complex) detection** (Shannon-energy envelope or template matching) → both fixed
  windows *and* an **RR-normalized, beat-aligned ensemble** (template + per-phase residual variance).
- **State vector** as conditioning metadata: HR, respiration rate, posture, activity (ENMO).
- **Channel-availability mask** `m ∈ {0,1}⁶` + **learned modality dropout**, so **3-channel
  SCG-only inference is in-distribution**, not three dead zero channels.

### Layer 2 — Foundation encoder
- **Overlapping multi-resolution conv stem** (kernels 25/50/125, stride 25) → **~96 tokens, not 20.**
  *(At 20 tokens / 5 visible under 75% masking, attention has no relational structure to learn.)*
- **Hierarchical Conformer** — D≈384, depth≈10, heads 6, RMSNorm, depthwise conv → ~18M params.
  *(Depth-4 / 3.3M is below where MAE yields semantic features for biosignals.)*
- **RoPE relative positions + a learned cardiac-phase embedding** (with phase-dropout fallback) —
  the single highest-leverage domain inductive bias: lets the model represent "same fiducial across
  beats."
- **Separate SCG/aux stems + cross-attention fusion + a learned channel gate** — decouples dead gyro
  from SCG. *(Also fixes the current fusion bug where both streams collapse to identical after
  layer 1.)*
- **Attention-pool head** (learned CLS query) + statistical tokens → the global feature. *(Mean-pool
  averages away the very HRV/morphology the twin should track.)*
- **Hybrid pretraining objective, not pure MAE** — span/beat + frequency masking with a
  **multi-resolution STFT loss** (so AO micro-vibrations aren't drowned by the systolic low-frequency
  term) + **data2vec-2.0 EMA latent regression** + **TS2Vec hierarchical contrastive** +
  **next-beat forecasting.** *(Periodicity makes random-patch MAE trivially solvable.)*
- **Transfer-first** — SSL-pretrain on public ECG/PPG/accelerometry (PTB-XL, MIMIC-IV-ECG, CODE-15,
  VitalDB, Capture-24) → **cross-modal ECG→SCG phase distillation** → LoRA/adapter adaptation on
  Digital Heart Twin. A *prerequisite*, not optional, to go past ~5–10M params on a small single-site set.

### Layer 3 — Personalization & anomaly (per user) — *the core innovation, upgraded*
- **Density, not posterior-shift** — replace KL-to-pooled-Gaussian with a **population-prior
  conditional Normalizing Flow** `p(h | c)`; anomaly score = `−log p`. Prototype/memory-bank kNN as
  cold-start fallback. *(The current VAE throws away decoder reconstruction error — the part that
  detects off-manifold inputs — and falls into the classic "VAE assigns high likelihood to OOD"
  trap.)*
- **Context conditioning** on `c = [activity, posture, HR/HRV, time-of-day, days-since-baseline,
  device, placement]`. Cheap path first: **Mondrian conformal strata**; graduate to the flow's
  context-net. **This is THE real-world viability blocker** — a workout must read as
  "normal-given-exercising."
- **Per-user conformal thresholds** at an explicit α (e.g. ~1 false alarm/month) — replaces the
  magic global `2.0`, whose true false-alarm rate differs for every user because KL units scale with
  each user's `baseline_var`.
- **Two-timescale drift tracker** (EWMA/Kalman random walk on baseline params, half-life ~weeks),
  Huberized, **frozen during suspected change-points** — adapts to benign drift (aging, fitness)
  without masking real decline.
- **Sequential change-point** — CUSUM / Page-Hinkley (controlled ARL₀) + Bayesian Online
  Change-Point Detection on the context-residual surprisal, replacing the degree-1 `np.polyfit`
  trend.
- **Uncertainty + abstention** — deep ensemble of the tiny heads + the input-quality gate →
  `INSUFFICIENT_QUALITY`, excluded from FAR, trend, and baseline adaptation.
- **Mixture-of-normals** per user, so bimodal "normal" (supine/upright, AM/PM) isn't half-flagged.

### Layer 4 — Decisioning & product
- **De-scoped output contract** — emit `{STABLE, CHANGED, INSUFFICIENT_QUALITY}` with a
  non-removable non-diagnostic disclaimer. **Delete the literal "Risk of deterioration increased"
  string** — that single line auto-classifies the product as a regulated medical device.
- **"What changed" decomposition** (rate vs morphology vs timing), each tied to a specific gold
  standard — never an undifferentiated scalar.
- **Graded clinician-in-the-loop tiers** — clinician adjudications are the *only* source of real
  outcome labels and feed threshold recalibration.

### Layer 5 — Deployment
- **Split seam** — shared, signed, INT8-quantized encoder on the phone NPU/ANE (one ONNX →
  CoreML + TFLite); the tiny **FP32** twin head + baseline buffers on CPU, **trained on-device so raw
  SCG never leaves the phone.**
- **Checkpoint decomposition** — store the encoder once (content-addressed); twins become ~80 KB
  deltas keyed by encoder SHA. **Refuse to load a twin against a mismatched encoder hash.** *(Today
  every `twin_user_*.pt` redundantly embeds an identical full encoder.)*
- **Federated MAE + DP-SGD for the encoder only** (v2+); the personal head is never aggregated.
- **Determinism + lockfile + side-effect-free imports**; a model/data registry; PSI/KS drift
  monitoring on DP-aggregated statistics.

---

## 2. Resolved tensions (positions taken, not options listed)

- **T1 — On-device budget vs a bigger backbone.** *Pretrain big, ship quantized/distilled — never
  train small.* The 18M Conformer is justified **only** by transfer pretraining; it runs INT8 on the
  NPU at <30 ms/window. If Android NNAPI rejects attention, **distill to a depthwise-separable CNN
  student** rather than shrink the teacher.
- **T2 — Adaptive baselines vs regulatory change-control.** *Adapt, but as a bounded, pre-declared
  operation inside a Predetermined Change Control Plan.* The weeks-scale Huberized tracker that
  **freezes on change-points** is simultaneously the anti-poisoning safeguard *and* the regulatory
  containment. Never ship free-running adaptation; never ship a frozen baseline.
- **T3 — Richer models vs small-data overfitting.** *The representation is upper-bounded by
  transfer; the density head stays shrinkage-regularized.* Encoder earns capacity only via
  public-corpus transfer; the per-user density is population-prior warm-started (fine-tune 2–3
  coupling layers); at n=5–20 recordings the **prototype-bank fallback wins at cold-start.** Any
  complexity that doesn't beat a frozen-feature Mahalanobis control on leave-one-subject-out is
  deleted.
- **T4 — Context conditioning vs explaining away real disease (the quietest, most dangerous one).**
  Conditioning on HR/activity can **absorb a true arrhythmia.** *Keep an **unconditioned
  rhythm/morphology pathway** running in parallel with the context-conditioned density.* Context may
  explain away benign nuisance; it must **never** be allowed to explain away rhythm or morphology
  anomalies. Non-negotiable.
- **T5 — Gate aggressively (lose data) vs retain data (small set).** *Gate consistently in **both**
  baseline-build and inference*; treat rejected windows as `abstain`, not as evidence of health. A
  baseline/scoring distribution mismatch (silent threshold drift) costs far more than lost windows.

---

## 3. Phased roadmap with gates

### P0 — Make the existing pipeline trustworthy & reproducible *(no architecture change)*
1. **Leakage-audit CI gate + full determinism** — GroupKFold by subject, window-id provenance,
   `use_deterministic_algorithms`, lockfile, side-effect-free `config.py`.
2. **De-leak EER** — disjoint enrollment/test by session/day, leave-one-subject-out impostors, all
   windows, subject-level bootstrap CIs.
3. **De-circularize early-warning** — split the simulator into independent `DiseaseProcess` +
   `NuisanceProcess`; report severity-sweep pAUC(FPR≤0.05) + AUPRC-at-prevalence; the disease curve
   must rise while the nuisance-only curve stays near chance.
4. **Per-user conformal threshold** replacing the global 2.0 — works on *today's* KL score
   immediately.
5. **Baseline gauntlet** — HR/HRV, SQI, population model, feature-Mahalanobis; paired DeLong tests
   with Benjamini–Hochberg FDR.
6. **De-scope the output contract** — delete the diagnostic string; add disclaimer +
   `INSUFFICIENT_QUALITY`.

**Exit:** CI fails on any leakage; metrics reproduce bit-identically across seeds; the twin **beats
every trivial baseline with a paired CI excluding 0**, or the null is reported honestly.
**Biggest risk:** the frozen MAE feature may simply not encode disease morphology — it caps the whole
stack. *De-risk:* a probe/kNN/effective-rank harness quantifies this **before** P2 investment.

### P1 — Real-data retrospective validation
1. **Label-free signal fixes first** — MAD norm, unit/rate calibration, monotonic resampling,
   NaN/clip/flatline guards.
2. **SQI module + hard gate** wired into dataset, HealthMonitor, *and* benchmark (consistently).
3. **Honest gyro path** — channel mask + modality dropout; split out the real respiration channel;
   stop feeding zeros.
4. **Rung-2 evidence on the real Zenodo data** — biometric-verification EER (no longitudinal
   outcomes needed) + any in-cohort label contrast, with bootstrap CIs.
5. **Analytical validity** — test-retest repeatability, sensor/phone reproducibility; **equity audit
   harness** (prespecified subgroups, FAR/sensitivity-gap bounds), documenting missing subgroup
   labels as a known limitation.
6. **Consent, data governance, audit logging** — preconditions for collecting anything real.

**Exit:** reproducible LOSO EER with subject-level CIs on real data; SQI gate demonstrably cuts
motion false-alarms; analytical repeatability characterized; **no clinical-efficacy claim asserted.**
**Biggest risk:** single-site data doesn't generalize across devices/bodies. *De-risk:* device/unit
harmonization + subgroup stratification surface the gap explicitly.

### P2 — Advanced architecture + context-conditioned personalization
1. **Representation-eval harness first** (probe / kNN / effective-rank / phase-decodability) —
   nothing below is measurable without it.
2. **Tokenization** (overlapping multi-res stem + RoPE + phase embedding), then
   **Conformer/RMSNorm/attention-pool** behind a config flag; **re-pretrain, never partial-load.**
3. **Hybrid SSL** added incrementally with collapse guards (data2vec → TS2Vec → next-beat), ablating
   each.
4. **Transfer + cross-modal distillation** (highest ceiling, highest cost) — validated against a
   from-scratch baseline before being trusted.
5. **Conditional flow density `p(h|c)`** warm-started from the population prior; **two-timescale
   drift tracker** coupled to the **CUSUM/BOCPD** change-point layer.
6. **Mixture-of-normals + uncertainty/abstention** ensemble.

**Exit:** each component clears the probe harness *and* the de-leaked benchmark with a paired CI gain;
per-user FAR matches the conformal α target; the context-conditioned score keeps the disease curve up
while flattening the nuisance curve.
**Biggest risk:** ECG→SCG domain gap makes transfer net-negative; SSL silently collapses. *De-risk:*
effective-rank/VICReg variance instrumented every epoch; transfer gated behind a from-scratch
ablation.

### P3 — On-device / federated + prospective clinical study
1. **ONNX → CoreML/TFLite INT8** within a quantization-error budget (rel L2 < 0.02, |ΔKL| < 0.1τ);
   **checkpoint decomposition + encoder-hash coupling.**
2. **On-device twin training** (Core ML `MLUpdateTask` / LiteRT signatures); raw SCG never leaves the
   device; battery/latency enforced via a DSP pre-gate + duty-cycling.
3. **Model/data registry, signed artifacts, PSI/KS drift monitoring** on DP-aggregated stats;
   clinician-in-the-loop tiers producing real labels.
4. **Federated MAE + DP-SGD** for the encoder only (v2+); measure the ε-vs-utility curve before
   promising anything.
5. **PCCP + IEC 62304 / ISO 14971 / ISO 13485** dossier; FDA De Novo / EU MDR path.
6. **Pre-registered prospective longitudinal cohort** with echo/NT-proBNP/Holter + adjudicated
   events; survival analysis (time-dependent AUC, Cox with time-varying covariate); **PPV/NPV at real
   prevalence.**

**Exit:** the quantized on-device twin reproduces the server KL within budget; the prospective
primary endpoint (sensitivity/specificity vs adjudicated outcome) is met with pre-registered CIs at
realistic prevalence.
**Biggest risk:** no longitudinal healthy→sick ground truth exists; DP-SGD may collapse encoder
utility. *De-risk:* the clinician-adjudication loop bootstraps the first real labels; DP utility
measured on a held-out cohort before any DP claim.

---

## 4. Top 10 highest-leverage moves (ranked by impact-vs-effort)

| # | Move | Effort | Dimension |
|---|------|:---:|---|
| 1 | **Per-user conformal threshold** replacing global 2.0 (works on today's score) | S/M | Personalization |
| 2 | **Leakage-audit CI + de-leaked EER + de-circularized early-warning** | S–M | Evaluation |
| 3 | **SQI hard gate before the encoder**, applied in baseline *and* inference | M | Signal |
| 4 | **Delete the diagnostic alert string + add `INSUFFICIENT_QUALITY` abstain** | S | Clinical |
| 5 | **Context conditioning as Mondrian conformal strata** (cheap viability fix) | L | Personalization |
| 6 | **Checkpoint decomposition + encoder-hash coupling + determinism/lockfile** | S | MLOps |
| 7 | **Conditional Normalizing Flow density** (score = −log p) | L | Personalization |
| 8 | **Cycle-aware tokenization** (overlapping multi-res stem + RoPE + phase embedding) | L | Foundation |
| 9 | **CUSUM/BOCPD change-point + two-timescale drift tracker** (replaces polyfit) | M | Personalization |
| 10 | **Transfer pretraining + ECG→SCG cross-modal distillation** (highest ceiling) | XL | Foundation |

The ranking is deliberately front-loaded with **cheap moves that make everything else measurable and
honest** (#1–#4, #6) before the expensive model upgrades. The personalization dimension dominates
because it sits between the encoder and the product, where impact-per-effort concentrates.

---

## 5. Hard open problems (success genuinely uncertain)

- **OP1 — The confounder/context problem.** Caffeine, exercise, posture, stress, illness, and sensor
  slip all move SCG morphology indistinguishably from cardiac change — *and* context is self-derived
  from the same signal, so derivation error leaks back as confounding. *De-risk:* validate the
  context estimators independently; Mondrian/weighted conformal stratified by context; **abstain on
  extreme/unmodeled context**; keep the unconditioned rhythm pathway (T4). **Residual uncertainty:
  high — this is the central viability bet.**
- **OP2 — No longitudinal ground truth.** Every clinical-sounding number today is circular. *De-risk:*
  a four-rung evidence ladder (never let a metric exceed the rung it passed); the
  clinician-in-the-loop loop is the *only* path to real labels and must be built in early;
  pre-register the prospective cohort and surface the multi-hundred-subject sample size up front.
  **Residual uncertainty: high and irreducible without years of data — honesty about it is part of
  the design.**
- **OP3 — Gyro absence.** Channels 4–6 are identically zero with no uncalibrated stream; the proxy
  physics is wrong. *De-risk:* channel mask + modality dropout (3-channel inference first-class);
  repurpose the residual as a correctly-labeled respiration channel; separate-stream encoding so real
  GCG drops in later with no re-architecture. **Residual uncertainty: low — clean engineering fix,
  but it caps available information until real gyro hardware ships.**
- **OP4 — Generalization from a small, single-site dataset.** Pure MAE at depth overfits device
  statistics; the frozen feature may not encode disease morphology, upper-bounding everything
  downstream. *De-risk:* transfer + cross-modal distillation as a prerequisite (validated against
  from-scratch); population-prior + empirical-Bayes shrinkage for the per-user density; the probe
  harness quantifies representation quality before over-investing. **Residual uncertainty:
  medium-high — transfer may be net-negative across the chest-phone-SCG vs clinical-ECG gap, which is
  why it stays gated behind an ablation.**

---

## The sequencing invariant that ties it together

> **Freeze the input contract → re-pretrain → re-fit twins → calibrate → evaluate — in that order,
> every time the DSP or encoder changes.**

Any normalization, filtering, tokenization, or `embed_dim` change invalidates every downstream
checkpoint and every per-user KL geometry. The **encoder-hash coupling** enforces this in production;
the **leakage-audit CI** enforces it in development. Violating this invariant is the single most
common way this system silently breaks.
