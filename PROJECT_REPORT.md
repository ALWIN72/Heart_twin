# MSCardio Digital Heart Twin — Technical Study & Review

*A full read-through and critical review of the codebase (~3,500 lines, ~30 Python files).
Findings were produced by reading every source file and cross-checked by a multi-agent review
that also **executed** `python run_pipeline.py --smoke_test` end-to-end.*

---

## 1. Executive summary

**MSCardio Digital Heart Twin** is a research prototype for **personalized cardiac anomaly
monitoring from smartphone seismocardiography (SCG)**. A self-supervised transformer
*foundation model* is pretrained (masked-autoencoder) on everyone's accelerometer/gyro signals;
then a lightweight **per-user VAE "personal twin"** sits on top of the frozen encoder and learns
what *that individual's* heartbeat normally looks like, flagging deviations from their **own**
baseline (not a population average) via a KL-divergence anomaly score.

The verdict, in one line: **the engineering is genuinely good and runnable, the core math is
correct, the documentation is unusually honest — but the headline benchmark numbers cannot yet be
trusted**, because Phase 5 evaluates the model on a self-referential synthetic loop and a handful
of real bugs silently degrade or bias the metrics.

| Dimension | Assessment |
|---|---|
| Code quality / structure | **Strong** — config-centralized, modular, clean |
| Runnability | **Verified** — smoke test completes all 5 phases + 8/8 unit tests pass |
| Core mathematics (KL / MAE / DANN / EER) | **Correct** — verified term-by-term |
| Documentation honesty | **Excellent** — scaffolding is openly labeled |
| Benchmark / scientific validity | **Weak** — circular evaluation, leakage, degenerate output |
| Production readiness | **Not close** — prototype scaffold |

---

## 2. What it is and how it's organized

A six-channel signal (3 SCG accelerometer axes + 3 gyroscope axes) flows through a staged
pipeline. Because the public MSCardio release ships only calibrated + uncalibrated SCG (no real
gyroscope), the last 3 channels are currently an explicit, clearly-labeled **"gyro proxy."**

```
Raw recordings (Subject_*/Recording_*/scg.csv, Uncalibrated_scg.csv)
   │
   ▼  Phase 1  Pretrain  — Masked-Autoencoder over the shared SCGEncoder (no labels)
   │
   ├── Phase 2  Denoiser    — SCG↔gyro cross-attention motion-artifact cancellation
   ├── Phase 3  Harmonizer   — device-invariant features via domain-adversarial (DANN) training
   └── Phase 3.5 Placement   — Sternum/Left/Right classifier + spatial (channel-mixing) corrector
   │
   ▼  Phase 4  Personal Heart Twin — per-user VAE head on the FROZEN encoder (the core innovation)
   │
   ▼  Phase 4.2 HealthMonitor — production inference: placement → harmonize → KL anomaly → alert
   │
   ├── Phase 5  Benchmark — EER, AUC-ROC, lead time, false-alarm rate
   └── Phase 6  Dashboard — Streamlit app
```

### Subsystem map

| Layer | Files | Role |
|---|---|---|
| Config | `src/config.py` | Central, *derived* constants (window=2500, patch=125, 20 patches, 6 ch) |
| Data | `src/data_loader.py`, `src/dataset.py` | Manifest building, subject-wise splits, CSV→(6, 2500) windows |
| Signal utils | `src/utils/preprocessing.py` | Resample, band-pass (0.8–30 Hz), windowing, **gyro proxy** |
| Fixtures | `src/utils/synthetic_data.py`, `src/utils/simulate_disease.py` | Dev-only data + deterioration simulator |
| Foundation | `src/models/encoder.py`, `fusion_transformer.py`, `mae.py` | ViT-style backbone, cross-modal fusion, MAE objective |
| Adaptation | `src/models/denoiser.py`, `dann.py`, `placement_classifier.py`, `spatial_transformer.py` | Denoise / harmonize / placement |
| Twin | `src/models/personal_twin.py`, `src/utils/anomaly_scorer.py` | Per-user VAE + production `HealthMonitor` |
| Training | `src/training/*.py` (5 loops) | One re-runnable loop per phase |
| Eval / app | `src/evaluation/benchmark.py`, `run_pipeline.py`, `src/dashboard.py` | Metrics, orchestration, UI |
| Tests | `tests/test_pipeline.py` | 8 shape/wiring smoke tests |

---

## 3. The core innovation, verified

The personal twin (`src/models/personal_twin.py`) is the heart of the project, and its math is
**correct**:

- The foundation encoder is frozen (`requires_grad=False`); a tiny VAE head (256→32→8 latent)
  is trained only on one user's recordings.
- `set_baseline(...)` aggregates a batch of known-healthy windows into a stored normal
  distribution `N(baseline_mu, baseline_var)`. Variance is pooled in **linear** space then
  re-logged — the statistically correct way (averaging log-variances would understate spread).
- `anomaly_score(x)` returns the closed-form KL divergence between the new window's posterior
  `q(z|x)=N(mu, exp(logvar))` and the stored baseline:

  ```
  KL = 0.5 · Σ [ exp(logvar − base_logvar) + (base_mu − mu)² / exp(base_logvar)
                 − 1 − (logvar − base_logvar) ]
  ```

  This matches the textbook diagonal-Gaussian KL exactly. It correctly **raises** if no baseline
  has been set — a contract the unit tests verify.

Likewise verified correct: the **MAE** masked-only reconstruction loss
(`(loss_per_patch * mask).sum() / mask.sum()`, `src/models/mae.py`), the **DANN** gradient
reversal (`src/models/dann.py`), and the **EER** decision function (negated score, FPR=FNR
crossing, `src/evaluation/benchmark.py`).

---

## 4. Strengths

1. **Correct core mathematics** — KL anomaly score, MAE loss, DANN reversal, and EER all check out
   term-by-term.
2. **Exceptional disclosure discipline** — synthetic data, the disease simulator, the gyro proxy,
   and the placement-label fallback are all explicitly labeled as stand-ins, with repeated
   "not a measure of real-world performance" caveats. This is rare and good scientific hygiene.
3. **Sound high-level design choices** — *subject-wise* splitting (not recording-wise) prevents
   identity leakage; per-user self-baseline scoring is the right framing for a signal with huge
   inter-person variability.
4. **Verified end-to-end runnability** — config-derived shapes keep data and model dimensions
   mutually consistent; the Phase-1→2/3/4 frozen-encoder handoff (`mae.export_encoder`) is
   key-consistent; the documented smoke test and all 8 unit tests pass (~9.5 s) on a fresh,
   *newer-than-pinned* environment (Python 3.14 / torch 2.10-cpu).
5. **Robust defensive engineering** — graceful degradation at every boundary (missing
   metadata / uncal / gyro, short recordings, odd CSV layouts, empty val splits, missing
   checkpoints → warn + random-init), `torch.load(weights_only=True)` everywhere, and
   `drop_last` / empty-loader guards.

---

## 5. Issues, ranked

> Findings are grouped into **real bugs** (defects in code), **limitations by design** (defensible
> choices), and **scientific caveats** (validity limits beyond any one bug). Severity reflects
> impact on the trustworthiness of results, not on whether the code crashes.

### 5.1 Real bugs

**[CRITICAL] `onset_day` is not forwarded into the deterioration simulator** —
`src/evaluation/benchmark.py` (≈line 140) vs `src/utils/simulate_disease.py` (≈line 23).
`evaluate_early_warning` labels days and computes lead-time against `symptom_onset_day` (default 20)
but calls `simulate_heart_deterioration(healthy_signal, days=...)` **without** passing `onset_day`,
so the simulator uses its own default (also 20). They agree **only by coincidence of matching
defaults**. Any sweep over onset — the natural early-warning experiment — silently labels against
an onset the signal does not exhibit, making AUC / lead-time / false-alarm-rate meaningless with
no error raised. (The dashboard *does* forward `onset_day` correctly, so the two callers disagree.)

**[HIGH] Cross-modal fusion collapses to self-attention after layer 1** —
`src/models/fusion_transformer.py` (`CrossAttentionFusionEncoder.forward`). After each fusion
layer, the merged output is projected and assigned to **both** streams
(`scg_tokens, gyro_tokens = half, half`), so from layer 2 onward the two inputs are byte-identical
and the "SCG and gyro attend to each other" mechanism degenerates into ordinary self-attention.
With the configured `fusion_layers=2`, only the first layer performs true cross-modal fusion — the
denoiser's advertised dual-stream "breakthrough" is half-defeated.

**[HIGH] EER train/eval leakage + first-batch-only undersampling** —
`src/evaluation/benchmark.py` (`evaluate_verification`). "Genuine" scores are taken over *all* of a
subject's recordings, **including the first 5 the twin's baseline was calibrated on**
(`train_personal_twin.py`). Genuine windows are therefore scored against a distribution built
partly from themselves, deflating genuine scores and thus EER. Separately, `next(iter(loader))`
scores only the first ≤20 windows per subject in non-shuffled order, discarding the rest. The
project even *has* the right tool — `create_splits` — but the benchmark does not use it.

**[HIGH] The production placement-correction stage is effectively a no-op** —
`src/utils/anomaly_scorer.py` + `src/models/spatial_transformer.py`. `SpatialCorrector` is
instantiated fresh, **never loaded from a checkpoint**, and `finetune_placement.py` never trains it
(no checkpoint is ever produced). At inference it is therefore a fixed *near-identity* channel mix,
and its output is then run through `harmonizer.harmonize` (an autoencoder reconstruction) that
would wash out any fine correction anyway. The advertised placement → correction step contributes
essentially nothing to the final score.

**[HIGH] A green smoke test hides a degenerate benchmark** —
`run_pipeline.py` / `src/evaluation/benchmark.py`. An actual smoke run prints `=== Done ===` and
exits 0 while writing **EER≈0.52, AUC≈0.66, Lead Time=NaN, FAR=0.0** to
`checkpoints/benchmark_results.json` as if they were stable results. Nothing warns that the run is
degenerate — a false-confidence trap.

**[MEDIUM] Window index/stride inconsistency** — `src/dataset.py`. `_build_index` always counts
windows non-overlappingly (`n_samples // window_samples`), but `_load_windows` passes `self.stride`
to `make_fixed_windows`. With an overlapping stride, `__len__` undercounts and `__getitem__`'s
`min(w, n-1)` clamp silently duplicates the last window. Length and contents can diverge from
reality.

**[MEDIUM] VAE training KL is ~8× weaker than nominal and inconsistent with scoring** —
`src/training/train_personal_twin.py`. The training KL uses `torch.mean(...)`, averaging over batch
**and** the 8 latent dims, so `kl_weight=1e-3` behaves like ~1.25e-4. Meanwhile `anomaly_score`
**sums** over the latent dim. The regularizer is weaker than intended and the train/score objectives
are mismatched, affecting baseline calibration.

**[MEDIUM] Non-deterministic validation → noisy "best" checkpoints** — `mae.py`, `pretrain.py`,
`finetune_denoiser.py`, `finetune_placement.py`. Random masking / motion injection / pretext labels
are re-drawn every eval pass regardless of `model.eval()`, so validation measures a *different task*
each epoch and a worse encoder can "win" on an easier random mask. No `cudnn.deterministic` /
`use_deterministic_algorithms` flags are set, and DataLoaders pass no generator — runs are not
bit-reproducible.

**[MEDIUM] The one data-pipeline test is non-hermetic** — `tests/test_pipeline.py`.
`test_data_pipeline` builds a tempdir dataset but then calls `create_splits` on a **hardcoded
repo-relative** `data/manifests/multimodal_manifest.csv`, validating whatever happens to be on disk
rather than the isolated dataset it just created. It passes today only by accidental filesystem
side effects.

**[MEDIUM] Per-user twins embed and strict-load the entire encoder** — `src/models/personal_twin.py`
holds the encoder as a submodule, so each `twin_user_*.pt` redundantly stores all ~53
`global_encoder.*` tensors and `load_user_twin` strict-loads them, overwriting the freshly-loaded
foundation. If `encoder_foundation.pt` is retrained without retraining twins, inference silently
uses a **stale** embedded encoder; any architecture change errors on load. A clean design would save
only the personal head + baseline buffers.

**[MEDIUM] Dead `num_devices` wiring collapses a 3rd platform into iOS** —
`src/training/finetune_harmonizer.py`. `n_platforms = max(2, nunique())` is computed but unused; the
DANN head is hardcoded to `num_devices=2` and unknown platforms map to class 0, so a third platform
is silently lumped in with iOS.

**[MEDIUM] Dashboard crashes on a baseline-less twin; off-by-one lead-time** — `src/dashboard.py`.
`analyze_recording` guards `twin is None` but not `has_baseline == False`, so a present-but-
uncalibrated twin raises an uncaught `RuntimeError`. The simulation lead-time also mixes a 0-based
flag index with a 1-based onset marker.

**[LOW] Misc smells** — unbounded in-memory dataset cache (never evicted); three duplicated
`patchify` helpers that risk silent divergence; `config.py` performs disk-writing `mkdir` as an
import side effect; relative `data/raw` defaults break when run outside the project root; docstrings
promise "loud warnings" / "dataset-level summary" logs for degenerate inputs that are never actually
emitted.

### 5.2 Limitations by design (defensible)

- **Gyro proxy = low-pass(uncal − cal)** keeps the 6-channel interface stable with a documented
  real-GCG swap path, rather than zero-padding. Reasonable engineering; only the *labeling* is the
  issue (see §5.3).
- **Self-supervised fallbacks** (synthetic motion injection, synthetic rotation pretext, synthetic
  disease) where labels are absent — honestly disclosed and the right call for an unlabeled release.
- **VAE reconstructs the frozen 256-d global feature** (not raw signal) — intentional and clean.

### 5.3 Scientific caveats (validity limits)

- **Circular Phase-5 evaluation.** The same simulator that injects "disease" also defines the
  labels, so a high AUC is near-tautological: it measures "did the score rise when the simulator
  turned on," not detection of real cardiac deterioration. This persists *even on real training
  data*, because the simulator is the only source of "deteriorated" samples.
- **The gyro proxy is physically mislabeled.** A gyroscope measures angular velocity; the
  low-frequency *linear*-acceleration residual is a different quantity. Worse, on real data lacking
  an uncalibrated stream, the code sets `uncal = cal.copy()`, making the proxy `lowpass(0)` = **three
  dead zero channels** silently fed into the "cross-modal" model.
- **"Lead Time > 14 days" is dimensionless.** Simulator "days" are a loop counter with no temporal
  spacing; the lead time is determined entirely by where `onset_day` sits relative to detector
  sensitivity.
- **`anomaly_threshold = 2.0` is mislabeled "2-sigma."** A KL value (in nats, always ≥ 0) has no
  sigma interpretation, and it is a single **global** constant — undercutting the "personalized"
  alerting claim exactly where personalization would matter (per-user FAR/lead-time).
- **Baseline variance pooling ignores between-sample spread**, under-estimating true baseline
  variance and thus systematically inflating anomaly scores / false positives.

---

## 6. README vs. code

The README is unusually accurate — every *major* advertised capability was confirmed in code, and
no claim is dangerously false. The discrepancies are localized attribution/behavior mismatches:

- **"The gyro-proxy swap happens in `build_multimodal_manifest` … the single place this swap
  happens"** — wrong location. `build_multimodal_manifest` only records a `gyro` *path*; the actual
  proxy-vs-real selection lives in `preprocessing.assemble_six_channel` (driven by whether
  `dataset.py` loaded a real `gyro.csv`).
- **"`data_loader.py` already checks for and prefers a real gyro column"** — the preference is in
  `dataset.py` / `preprocessing.py`, not `data_loader.py`, and keys off a separate `gyro.csv`
  *file*, not a column.
- **`finetune_placement.py` docstring mentions joint `SpatialCorrector` fine-tuning** — the
  corrector is never imported, instantiated, or trained there.
- **Docstrings promise warnings that never fire** — single-column CSV tiling claims "a loud
  warning"; the no-uncal degenerate path claims a "dataset-level summary." Neither log exists, so
  genuinely degenerate inputs (3 identical SCG axes; 3 zero gyro channels) pass silently.
- **"Foundation model reused by every downstream task"** — the placement classifier is a standalone
  CNN that gets zero benefit from Phase-1 pretraining.
- **`benchmark.py` lead-time comment says "negative = caught early"** — the code and the README
  target ("> 14") are positive-when-early; only the inline comment is inverted.

---

## 7. Empirical smoke-test result

`python run_pipeline.py --smoke_test` was executed end-to-end on Windows 11 / Python 3.14 /
torch 2.10-cpu (well ahead of the `>=` dependency floors):

- All 5 phases completed; every checkpoint was written; the frozen-encoder handoff worked.
- All 8 unit tests passed in ~9.5 s.
- The benchmark produced **degenerate** values (EER≈0.52, AUC≈0.66, **Lead Time=NaN**, FAR=0.0) —
  exactly what one expects from tiny synthetic data and an untrained-ish pipeline, but emitted with
  no warning. This is confirmation that the *plumbing* runs, not that detection works.

---

## 8. Verdict & recommendations

**This is a research prototype / engineering scaffold — not a working detector, and far from
production.** Its real, defensible contribution is the architecture and the disclosure discipline:
a clean, runnable, config-consistent multi-phase SCG pipeline with correct core math and honest
docs. Its weakness is that the reported metrics measure the test harness, not cardiac physiology.

To make the numbers trustworthy, in priority order:

1. **Fix the evaluation harness** — forward `onset_day` (and a fixed seed) into the simulator; make
   EER baseline/eval windows disjoint and score *all* windows via `create_splits`; have the
   benchmark fail loudly on NaN/degenerate metrics.
2. **Break the circular evaluation** — obtain real longitudinal or independently-labeled
   deterioration data so labels don't come from the same generator as the signal.
3. **Make the advertised pipeline real** — actually train/load `SpatialCorrector` (or drop the
   claim) and fix the fusion stream-collapse.
4. **Acquire real gyroscope channels** so the cross-modal premise isn't fed zero/duplicate channels.
5. **Reproducibility hardening** — pin dependencies + add a lockfile (and drop the unused `gradio`
   dependency); make validation deterministic; calibrate per-user thresholds instead of a global 2.0.

Until at least (1)–(4) are done, every Phase-5 figure should be reported as *"pipeline-executes
smoke output,"* not as a measurement of clinical performance.

---

*Files reviewed: `src/config.py`, `src/data_loader.py`, `src/dataset.py`,
`src/utils/{preprocessing,synthetic_data,simulate_disease,anomaly_scorer}.py`,
`src/models/{encoder,fusion_transformer,mae,denoiser,dann,placement_classifier,spatial_transformer,personal_twin}.py`,
`src/training/{pretrain,finetune_denoiser,finetune_harmonizer,finetune_placement,train_personal_twin}.py`,
`src/evaluation/benchmark.py`, `src/dashboard.py`, `run_pipeline.py`, `tests/test_pipeline.py`,
`requirements.txt`, `README.md`.*
