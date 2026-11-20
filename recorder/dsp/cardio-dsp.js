/**
 * dsp/cardio-dsp.js - In-Browser Seismocardiography Signal Processing
 * Preserves the exact DSP math from the original Digital Heart Twin prototype:
 * - 200 Hz resampling
 * - Refractory beat detection
 * - HRV: SDNN, RMSSD, pNN50, Poincaré SD1/SD2, spectral LF/HF
 * - Thoracic respiration drift
 * - Motion artifact detection & quality classification
 * - Effective sampling rate check (< 100 Hz flag)
 */

export function avg(arr) {
  if (!arr || !arr.length) return 0;
  return arr.reduce((acc, v) => acc + v, 0) / arr.length;
}

export function std(arr) {
  if (!arr || !arr.length || arr.length < 2) return 0;
  const m = avg(arr);
  return Math.sqrt(arr.reduce((acc, v) => acc + (v - m) * (v - m), 0) / arr.length);
}

export function median(arr) {
  if (!arr || !arr.length) return 0;
  const s = Array.from(arr).sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Zero-phase forward-backward 2nd-order biquad filter (filtfilt) */
function biquadFiltFilt(sig, b0, b1, b2, a1, a2) {
  const n = sig.length;
  if (n < 12) return Float64Array.from(sig);
  const run = (inp) => {
    const out = new Float64Array(n);
    let x1 = inp[0], x2 = inp[0], y1 = 0, y2 = 0;
    for (let i = 0; i < n; i++) {
      const x0 = inp[i];
      const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      out[i] = Number.isFinite(y0) ? y0 : 0;
      x2 = x1; x1 = x0;
      y2 = y1; y1 = out[i];
    }
    return out;
  };
  const fwd = run(sig);
  const revIn = new Float64Array(n);
  for (let i = 0; i < n; i++) revIn[i] = fwd[n - 1 - i];
  const revOut = run(revIn);
  const res = new Float64Array(n);
  for (let i = 0; i < n; i++) res[i] = revOut[n - 1 - i];
  return res;
}

/**
 * Zero-phase 4-25 Hz Cardiac Valve Bandpass Filter (Butterworth HP + LP cascaded).
 * Strips <3.5 Hz respiratory chest expansion and >25 Hz sensor/muscle noise,
 * isolating pure Aortic Opening (AO) & Closure (AC) mechanical bursts.
 */
export function bandpassCardiac4_25(sig, fs = 200) {
  const n = sig.length;
  if (n < 16 || fs < 20) {
    const m = avg(Array.from(sig));
    return Float64Array.from(sig, (v) => v - m);
  }
  const m0 = avg(Array.from(sig));
  const centered = Float64Array.from(sig, (v) => v - m0);

  // 1. 2nd-order Butterworth High-Pass at 4.0 Hz
  const fHp = Math.min(4.0, fs * 0.15);
  const kHp = Math.tan((Math.PI * fHp) / fs);
  const kHp2 = kHp * kHp;
  const sq2 = Math.SQRT2;
  const normHp = 1 / (1 + sq2 * kHp + kHp2);
  const hp = biquadFiltFilt(
    centered,
    normHp,
    -2 * normHp,
    normHp,
    2 * (kHp2 - 1) * normHp,
    (1 - sq2 * kHp + kHp2) * normHp
  );

  // 2. 2nd-order Butterworth Low-Pass at min(25.0 Hz, 0.42 * fs)
  const fLp = Math.min(25.0, fs * 0.42);
  if (fLp <= fHp + 2) return hp;
  const kLp = Math.tan((Math.PI * fLp) / fs);
  const kLp2 = kLp * kLp;
  const normLp = 1 / (1 + sq2 * kLp + kLp2);
  return biquadFiltFilt(
    hp,
    kLp2 * normLp,
    2 * kLp2 * normLp,
    kLp2 * normLp,
    2 * (kLp2 - 1) * normLp,
    (1 - sq2 * kLp + kLp2) * normLp
  );
}

export function decim(arr, n) {
  const out = [];
  const L = arr.length;
  if (!L) return out;
  for (let i = 0; i < n; i++) {
    out.push(arr[Math.min(L - 1, Math.floor((i / (n - 1)) * (L - 1)))]);
  }
  return out;
}

/** Resample tri-axial motion time series to exactly fs Hz (200 Hz default) */
export function resampleMag(rows, fs = 200) {
  if (!rows || rows.length < 4) return new Float64Array(0);
  const t = rows.map((r) => r[0]);
  const mag = rows.map((r) => Math.hypot(r[1], r[2], r[3]));
  const t0 = t[0];
  const t1 = t[t.length - 1];
  const duration = Math.max(0.1, t1 - t0);
  const n = Math.max(4, Math.floor(duration * fs));
  const out = new Float64Array(n);
  let j = 0;

  for (let i = 0; i < n; i++) {
    const tt = t0 + i / fs;
    while (j < t.length - 2 && t[j + 1] < tt) j++;
    const a = t[j];
    const b = t[j + 1] ?? a + 1e-6;
    const fr = Math.min(1, Math.max(0, (tt - a) / (b - a || 1e-6)));
    out[i] = mag[j] * (1 - fr) + (mag[j + 1] ?? mag[j]) * fr;
  }
  return out;
}

/**
 * Centered Moving Average (Helper)
 * O(N) using prefix sums.
 */
export function movingAverageCentered(arr, w) {
  const n = arr.length;
  const h = Math.floor(w / 2);
  const prefix = new Float64Array(n + 1);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) prefix[i + 1] = prefix[i] + arr[i];
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - h);
    const b = Math.min(n, i + h + 1);
    out[i] = (prefix[b] - prefix[a]) / (b - a);
  }
  return out;
}

/**
 * Normalized Autocorrelation Periodicity with Octave / Sub-Harmonic Guard
 */
export function calculatePeriodicity(env, minLag, maxLag) {
  const len = env.length;
  if (!len) return { strength: 0, lag: 0 };
  let sum = 0;
  for (let i = 0; i < len; i++) sum += env[i];
  const m = sum / len;
  const centered = new Float64Array(len);
  let ac0 = 0;
  for (let i = 0; i < len; i++) {
    const v = env[i] - m;
    centered[i] = v;
    ac0 += v * v;
  }
  if (!ac0) return { strength: 0, lag: 0 };

  let bestLag = -1;
  let maxCorr = -1;
  const clampedMax = Math.min(maxLag, len - 2);
  const estFs = maxLag / 1.5;
  const corr = new Float64Array(len);
  for (let lag = minLag; lag <= clampedMax && lag < len; lag++) {
    let s = 0;
    for (let i = 0; i + lag < len; i++) {
      s += centered[i] * centered[i + lag];
    }
    const normCorr = s / ac0;
    corr[lag] = normCorr;
    if (normCorr > maxCorr) {
      maxCorr = normCorr;
      bestLag = lag;
    }
  }

  // Sub-harmonic guard: if bestLag is a 2x period repeat (half-BPM), prefer the fundamental lag/2 when strong
  if (bestLag > 0) {
    const halfCenter = Math.round(bestLag / 2);
    if (halfCenter >= minLag + 1) {
      let halfBest = -1;
      let halfMax = -1;
      for (let l = Math.max(minLag, halfCenter - 3); l <= Math.min(clampedMax, halfCenter + 3); l++) {
        if (corr[l] > halfMax) {
          halfMax = corr[l];
          halfBest = l;
        }
      }
      if (halfBest > 0 && halfMax >= 0.80 * maxCorr && halfMax >= 0.22) {
        bestLag = halfBest;
        maxCorr = halfMax;
      }
    }

    // S1+S2 double-counting guard: if bestLag > 100 BPM (<0.60*fs) and 2*bestLag aligns S1-to-S1
    // with >= 92% of peak correlation, prefer the true fundamental 2*bestLag (50-90 BPM)
    const dblCenter = bestLag * 2;
    if (bestLag < 0.60 * estFs && dblCenter <= clampedMax) {
      let dblBest = -1;
      let dblMax = -1;
      for (let l = Math.max(minLag, dblCenter - 4); l <= Math.min(clampedMax, dblCenter + 4); l++) {
        if (corr[l] > dblMax) {
          dblMax = corr[l];
          dblBest = l;
        }
      }
      if (dblBest > 0 && dblMax >= 0.92 * maxCorr) {
        bestLag = dblBest;
        maxCorr = dblMax;
      }
    }
  }

  return { strength: Math.max(0, maxCorr), lag: bestLag };
}

/**
 * Fuse SCG (accelerometer) and GCG (gyroscope) 4-25 Hz cardiac energy envelopes.
 * Gyrocardiography (GCG) often yields 2-3x sharper aortic opening peaks on soft surfaces.
 */
export function buildFusedCardiacEnvelope(scgSig, gyroSig, fs) {
  const hpScg = bandpassCardiac4_25(scgSig, fs);
  const wEnv = Math.max(2, Math.round(fs * 0.08));
  const sqScg = new Float64Array(hpScg.length);
  for (let i = 0; i < hpScg.length; i++) sqScg[i] = hpScg[i] * hpScg[i];
  const envScg = movingAverageCentered(sqScg, wEnv);

  if (!gyroSig || gyroSig.length !== scgSig.length) {
    return { env: envScg, hpScg, usedGyro: false };
  }

  const hpGyro = bandpassCardiac4_25(gyroSig, fs);
  const gyroStd = std(hpGyro);
  if (gyroStd < 1e-4) {
    return { env: envScg, hpScg, usedGyro: false };
  }

  const sqGyro = new Float64Array(hpGyro.length);
  for (let i = 0; i < hpGyro.length; i++) sqGyro[i] = hpGyro[i] * hpGyro[i];
  const envGyro = movingAverageCentered(sqGyro, wEnv);

  const scgScale = Math.max(1e-9, avg(Array.from(envScg)));
  const gyroScale = Math.max(1e-9, avg(Array.from(envGyro)));
  const fused = new Float64Array(envScg.length);
  for (let i = 0; i < envScg.length; i++) {
    fused[i] = 0.65 * (envScg[i] / scgScale) + 0.35 * (envGyro[i] / gyroScale);
  }
  return { env: fused, hpScg, usedGyro: true };
}

/**
 * Shared Smartwatch-Grade Consensus BPM & HRV Extractor
 * Combines harmonic-checked autocorrelation + adaptive-refractory peak detection + median RR.
 */
export function extractConsensusBpm(env, fs) {
  const per = calculatePeriodicity(env, Math.max(2, Math.floor(0.33 * fs)), Math.floor(1.50 * fs));
  const bpmAuto = per.lag > 0 ? (60 * fs) / per.lag : 0;

  const envArr = Array.from(env);
  const me = avg(envArr);
  const envSd = std(envArr);
  const thr = me + 0.55 * envSd;

  // Adaptive refractory period scaled by autocorrelation rate (prevents S2 double-counting & supports high HR)
  const adaptiveRefrSec = bpmAuto >= 45 && bpmAuto <= 180
    ? Math.max(0.27, Math.min(0.42, 0.52 * (60 / bpmAuto)))
    : 0.33;
  const refr = Math.max(2, Math.floor(adaptiveRefrSec * fs));

  const beats = [];
  let last = -refr;
  for (let i = 1; i < env.length - 1; i++) {
    if (env[i] > thr && env[i] >= env[i - 1] && env[i] > env[i + 1] && i - last >= refr) {
      beats.push(i);
      last = i;
    }
  }

  const rawRr = [];
  for (let i = 1; i < beats.length; i++) {
    const dtMs = ((beats[i] - beats[i - 1]) / fs) * 1000;
    if (dtMs >= 330 && dtMs <= 1500) rawRr.push(dtMs);
  }

  // Outlier-rejected RR intervals (within +-35% of median RR, like smartwatch PPG/ECG filters)
  let rr = rawRr;
  if (rawRr.length >= 3) {
    const med = median(rawRr);
    const clean = rawRr.filter((v) => Math.abs(v - med) / (med || 1) <= 0.35);
    if (clean.length >= 2) rr = clean;
  }

  const medRr = rr.length ? median(rr) : 0;
  const bpmBeats = medRr > 0 ? 60000 / medRr : 0;

  let bpm = 0;
  if (bpmAuto > 0 && bpmBeats > 0 && Math.abs(bpmAuto - bpmBeats) < 20) {
    bpm = Math.round(bpmAuto * 0.45 + bpmBeats * 0.55);
  } else if (per.strength >= 0.25 && bpmAuto > 0) {
    bpm = Math.round(bpmAuto);
  } else {
    bpm = Math.round(bpmAuto || bpmBeats);
  }

  return {
    bpm,
    bpmAuto: Math.round(bpmAuto),
    bpmBeats: Math.round(bpmBeats),
    strength: Math.max(0, Math.min(1, per.strength)),
    beats,
    rr,
  };
}

/** Centered Moving Average */
export function movavgC(x, w) {
  return movingAverageCentered(x, w);
}

/** Autocorrelation periodicity for cardiac rhythm */
export function periodicity(env, fs, lo, hi) {
  const m = avg(Array.from(env));
  const e = env.map((v) => v - m);
  let ac0 = 0;
  for (const v of e) ac0 += v * v;
  ac0 = ac0 || 1e-9;
  let best = -1;
  let bv = -1;
  for (let lag = lo; lag <= hi && lag < e.length; lag++) {
    let s = 0;
    for (let i = 0; i + lag < e.length; i++) s += e[i] * e[i + lag];
    const nv = s / ac0;
    if (nv > bv) {
      bv = nv;
      best = lag;
    }
  }
  return { strength: Math.max(0, bv), lag: best };
}

/** In-place Radix-2 FFT */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const c = Math.cos(ang * k);
        const s = Math.sin(ang * k);
        const pr = re[i + k + len / 2] * c - im[i + k + len / 2] * s;
        const pi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
        const ur = re[i + k];
        const ui = im[i + k];
        re[i + k] = ur + pr;
        im[i + k] = ui + pi;
        re[i + k + len / 2] = ur - pr;
        im[i + k + len / 2] = ui - pi;
      }
    }
  }
}

/** HRV Poincaré and spectral metrics */
export function hrvLab(rr) {
  if (!rr || rr.length < 4) return null;
  const mean = avg(rr);
  const d = [];
  for (let i = 1; i < rr.length; i++) d.push(rr[i] - rr[i - 1]);
  const sdsd = Math.sqrt(avg(d.map((v) => (v - avg(d)) ** 2)));
  const sdnn = Math.sqrt(avg(rr.map((v) => (v - mean) ** 2)));
  const sd1 = Math.SQRT1_2 * sdsd;
  const sd2 = Math.sqrt(Math.max(0, 2 * sdnn * sdnn - 0.5 * sdsd * sdsd));

  const fsr = 4;
  const tcum = [];
  let acc = 0;
  for (const v of rr) {
    acc += v / 1000;
    tcum.push(acc);
  }
  const dur = tcum[tcum.length - 1] || 1;
  const N = Math.max(16, 1 << Math.ceil(Math.log2(Math.max(2, dur * fsr))));
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  let j = 0;

  for (let i = 0; i < N; i++) {
    const t = i / fsr;
    while (j < tcum.length - 2 && tcum[j + 1] < t) j++;
    const a = tcum[j] || 0;
    const b = tcum[j + 1] || a + 1e-6;
    const fr = Math.min(1, Math.max(0, (t - a) / (b - a)));
    re[i] = (rr[j] || mean) * (1 - fr) + (rr[j + 1] || rr[j] || mean) * fr;
  }
  const mu = avg(Array.from(re));
  for (let i = 0; i < N; i++) re[i] -= mu;

  fft(re, im);
  let lf = 0;
  let hf = 0;
  for (let i = 1; i < N / 2; i++) {
    const f = (i * fsr) / N;
    const p = re[i] * re[i] + im[i] * im[i];
    if (f >= 0.04 && f < 0.15) lf += p;
    else if (f >= 0.15 && f < 0.4) hf += p;
  }

  return {
    sd1: Math.round(sd1),
    sd2: Math.round(sd2),
    lfhf: hf > 0 ? +(lf / hf).toFixed(2) : 0,
    rr,
  };
}

/**
 * Full signal analysis pipeline on raw sensor recordings:
 * rows: array of [timestamp, x, y, z] (linear SCG)
 * gyroRows: optional array of [timestamp, beta, gamma, alpha] (GCG)
 */
export function analyzeRecording(rows, gyroRows = null) {
  if (!rows || rows.length < 10) return { ok: false, error: 'Not enough data' };

  const duration = rows[rows.length - 1][0] - rows[0][0];
  const effectiveHz = duration > 0 ? Math.round(rows.length / duration) : 0;
  const isRateLimited = effectiveHz < 100; // e.g. iOS Safari 60 Hz limit

  const fs = 200;
  const sig = resampleMag(rows, fs);
  if (sig.length < fs * 3) return { ok: false, error: 'Recording too short (under 3 seconds)' };

  const gyroSig = gyroRows && gyroRows.length >= 10 ? resampleMag(gyroRows, fs) : null;
  const alignedGyro = gyroSig && gyroSig.length === sig.length ? gyroSig : null;

  const drift = movavgC(sig, Math.round(fs));
  const { env, hpScg, usedGyro } = buildFusedCardiacEnvelope(sig, alignedGyro, fs);
  const consensus = extractConsensusBpm(env, fs);

  const { bpm, beats, rr, strength } = consensus;
  const sdnn = rr.length > 1 ? Math.sqrt(avg(rr.map((v) => (v - avg(rr)) ** 2))) : 0;
  const dd = [];
  for (let i = 1; i < rr.length; i++) dd.push(rr[i] - rr[i - 1]);
  const rmssd = dd.length ? Math.sqrt(avg(dd.map((v) => v * v))) : 0;
  const pnn50 = dd.length
    ? (100 * dd.filter((v) => Math.abs(v) > 50).length) / dd.length
    : 0;

  const resp = calculatePeriodicity(drift, Math.floor(2 * fs), Math.floor(10 * fs));
  const respRate = resp.lag > 0 ? (60 * fs) / resp.lag : 0;

  let quality = 'Poor';
  let qclass = 'badge-diff';

  if (strength > 0.42 && beats.length >= 6) {
    quality = 'Good';
    qclass = 'badge-normal';
  } else if (strength > 0.22 && beats.length >= 4) {
    quality = 'Fair';
    qclass = 'badge-slight';
  }

  let irregular = false;
  if (rr.length >= 5) {
    const fr =
      rr.slice(1).filter((v, i) => Math.abs(v - rr[i]) / (rr[i] || 1) > 0.2).length /
      (rr.length - 1);
    irregular = fr > 0.3;
  }

  const hrvMetrics = hrvLab(rr);

  return {
    ok: true,
    fs,
    duration: Math.round(duration),
    effectiveHz,
    isRateLimited,
    usedGyro,
    bpm,
    beats,
    rr,
    sdnn: Math.round(sdnn),
    rmssd: Math.round(rmssd),
    pnn50: +pnn50.toFixed(0),
    resp: Math.round(respRate),
    strength,
    quality,
    qclass,
    irregular,
    hrv: hrvMetrics,
    respWave: decim(Array.from(drift), 160),
    hpWave: decim(Array.from(hpScg), 300),
  };
}

/** Check if reading deviates from personal baseline */
export function compareWithBaseline(metrics, baseline) {
  if (!baseline || !metrics || !metrics.bpm) {
    return {
      status: 'Early estimate',
      verdict: 'Normal for you',
      ratio: 0.8,
      isBaselineComparison: false,
    };
  }

  // Feature distance: BPM deviation, SDNN deviation
  const d_bpm = Math.abs(metrics.bpm - (baseline.bpm || metrics.bpm)) / (baseline.bpm || 70);
  const d_sdnn = Math.abs(metrics.sdnn - (baseline.sdnn || metrics.sdnn)) / (baseline.sdnn || 40);
  const ratio = (d_bpm * 1.5 + d_sdnn) / 1.8;

  let verdict = 'Normal for you';
  if (ratio > 1.5) {
    verdict = 'Different from your usual';
  } else if (ratio >= 1.0) {
    verdict = 'Slightly different';
  }

  return {
    status: verdict,
    verdict,
    ratio: +ratio.toFixed(2),
    isBaselineComparison: true,
  };
}

/**
 * Fast Live BPM on a rolling 8-second buffer of [[t_sec, mag], ...]
 * Matches Mobile_seismograph liveBPM() implementation.
 */
export function computeLiveBPM(liveBuf) {
  if (!liveBuf || liveBuf.length < 50) return 0;
  const sig = liveBuf.map((p) => p[1]);
  const gyroSig = liveBuf[0] && liveBuf[0].length >= 3 ? liveBuf.map((p) => p[2]) : null;
  const ts = liveBuf.map((p) => p[0]);
  const fs = sig.length / Math.max(0.001, ts[ts.length - 1] - ts[0]);
  const { env } = buildFusedCardiacEnvelope(sig, gyroSig, fs);
  return extractConsensusBpm(env, fs).bpm;
}

/**
 * Real-Time Signal Quality Index (SQI) & Placement Coach Algorithm
 * Evaluates live 8-second rolling magnitude buffer every 250 ms (4 Hz) using:
 * - Zero-phase 4-25 Hz Butterworth cardiac bandpass (eliminates breathing drift)
 * - SCG + Gyroscope (GCG) energy envelope fusion
 * - Harmonic-checked autocorrelation + adaptive-refractory median-RR consensus
 * - Gravity-vector orientation & motion artifact gating
 */
export function updateCoach(liveBuf, g = { x: 0, y: 0, z: 9.81 }) {
  if (!liveBuf || liveBuf.length < 20) {
    return {
      score: 0,
      badge: 'Calibrating',
      message: 'Calibrating sensors...',
      hint: 'Lie still on your back and rest phone on lower sternum.',
      color: '#9CA3AF',
      fs: 200,
      bpm: 0,
      isValidCardiacBeat: false,
      placement: 'Sternum',
      offsetX: 0,
      offsetY: 0,
      isOptimal: false,
    };
  }

  const mags = liveBuf.map((p) => p[1]);
  const gyroMags = liveBuf[0] && liveBuf[0].length >= 3 ? liveBuf.map((p) => p[2]) : null;
  const ts = liveBuf.map((p) => p[0]);
  const dur = Math.max(0.001, ts[ts.length - 1] - ts[0]);
  const fs = mags.length / dur; // Calculate true dynamic sampling frequency

  // 1. 4-25 Hz Cardiac Bandpass & SCG+Gyro Envelope Fusion
  const { env, hpScg } = buildFusedCardiacEnvelope(mags, gyroMags, fs);
  const sd = std(hpScg);

  // 2. Jerk / Motion artifact calculation on bandpass-filtered cardiac signal
  let jerk = 0;
  for (let i = 1; i < hpScg.length; i++) jerk += Math.abs(hpScg[i] - hpScg[i - 1]);
  jerk /= Math.max(1, hpScg.length - 1);

  // 3. Harmonic-checked Autocorrelation + Median-RR Consensus
  const consensus = extractConsensusBpm(env, fs);
  const strength = consensus.strength; // [0.0, 1.0]

  // Orientation analysis from gravity vector (Math.abs(gz) handles iOS negative Z)
  const normG = Math.hypot(g.x, g.y, g.z) || 9.81;
  const gx = g.x / normG;
  const gy = g.y / normG;
  const gz = Math.abs(g.z) / normG;

  const pitchAngle = Math.asin(Math.max(-1, Math.min(1, gy))) * (180 / Math.PI);
  const rollAngle = Math.atan2(gx, Math.max(0.05, gz)) * (180 / Math.PI);
  const offsetX = Math.max(-1, Math.min(1, gx * 2.0));
  const offsetY = Math.max(-1, Math.min(1, gy * 2.0));

  let placement = 'Center Sternum';
  if (gx > 0.45 || rollAngle > 28) {
    placement = 'Tilted Right';
  } else if (gx < -0.45 || rollAngle < -28) {
    placement = 'Tilted Left (Apex)';
  }

  // 4. Decision Tree for Guidance & SQI
  let score, badge, message, hint, color;
  if (sd < 0.002) {
    score = 12;
    badge = 'No Signal';
    message = 'No heartbeat detected';
    hint = 'Place phone firmly on the center of your chest.';
    color = '#EF4444';
  } else if (strength > 0.42) {
    score = 94;
    badge = 'Optimal Signal';
    message = 'Clear heartbeat detected';
    hint = 'Phone is stabilized over the sternum. Ready to record!';
    color = '#10B981';
  } else if (jerk > sd * 1.15 || sd > 0.45) {
    score = 35;
    badge = 'Motion Detected';
    message = 'Excessive movement';
    hint = 'Relax shoulders and breathe gently.';
    color = '#EF4444';
  } else if (strength > 0.22) {
    score = 62;
    badge = 'Fair Signal';
    message = 'Faint beat detected';
    hint = 'Slide phone slightly toward your lower breastbone.';
    color = '#F59E0B';
  } else {
    score = 28;
    badge = 'Weak Signal';
    message = 'Weak signal';
    hint = 'Lie flat on your back with the phone flat on your chest.';
    color = '#EF4444';
  }

  const isOptimal = score >= 62;
  const rawBpm = consensus.bpm;

  // Report BPM only when valid cardiac signal is present (not No Signal, not Motion Detected)
  const isValidCardiacBeat =
    sd >= 0.002 &&
    badge !== 'Motion Detected' &&
    strength >= 0.18 &&
    rawBpm >= 40 &&
    rawBpm <= 180;

  const bpm = isValidCardiacBeat ? rawBpm : 0;

  return {
    score,
    badge,
    message,
    hint,
    color,
    fs,
    bpm,
    isValidCardiacBeat,
    placement,
    isOptimal,
    offsetX,
    offsetY,
    pitchAngle,
    rollAngle,
  };
}

/**
 * Backward compatibility wrapper for assessRealtimePlacement
 */
export function assessRealtimePlacement(x, y, z, recentSamples = []) {
  const liveBuf = recentSamples.map((s) => [s[0], Math.hypot(s[1], s[2], s[3])]);
  const coach = updateCoach(liveBuf, { x, y, z });
  return {
    placement: coach.placement,
    quality: coach.score >= 80 ? 'Good' : coach.score >= 50 ? 'Fair' : 'Poor',
    score: coach.score,
    guidance: coach.hint,
    isOptimal: coach.isOptimal,
    offsetX: coach.offsetX,
    offsetY: coach.offsetY,
    pitchAngle: coach.pitchAngle,
    rollAngle: coach.rollAngle,
    isExcessMotion: coach.badge === 'Motion Detected',
  };
}

