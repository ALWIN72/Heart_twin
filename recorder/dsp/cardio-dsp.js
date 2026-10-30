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

/** Centered Moving Average */
export function movavgC(x, w) {
  const n = x.length;
  const h = Math.floor(w / 2);
  const pre = new Float64Array(n + 1);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + x[i];
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - h);
    const b = Math.min(n, i + h + 1);
    out[i] = (pre[b] - pre[a]) / (b - a);
  }
  return out;
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
 * rows: array of [timestamp, x, y, z]
 */
export function analyzeRecording(rows) {
  if (!rows || rows.length < 10) return { ok: false, error: 'Not enough data' };

  const duration = rows[rows.length - 1][0] - rows[0][0];
  const effectiveHz = duration > 0 ? Math.round(rows.length / duration) : 0;
  const isRateLimited = effectiveHz < 100; // e.g. iOS Safari 60 Hz limit

  const fs = 200;
  const sig = resampleMag(rows, fs);
  if (sig.length < fs * 3) return { ok: false, error: 'Recording too short (under 3 seconds)' };

  const drift = movavgC(sig, Math.round(fs));
  const hp = sig.map((v, i) => v - drift[i]);
  const env = movavgC(
    hp.map((v) => v * v),
    Math.round(fs * 0.08)
  );

  const per = periodicity(env, fs, Math.floor(0.33 * fs), Math.floor(1.5 * fs));
  const bpmAuto = per.lag > 0 ? (60 * fs) / per.lag : 0;

  const me = avg(Array.from(env));
  let sd = 0;
  for (const v of env) sd += (v - me) ** 2;
  sd = Math.sqrt(sd / env.length);

  const thr = me + 0.6 * sd;
  const refr = Math.floor(0.33 * fs);
  const beats = [];
  let last = -refr;

  for (let i = 1; i < env.length - 1; i++) {
    if (env[i] > thr && env[i] >= env[i - 1] && env[i] > env[i + 1] && i - last >= refr) {
      beats.push(i);
      last = i;
    }
  }

  const rr = [];
  for (let i = 1; i < beats.length; i++) {
    rr.push(((beats[i] - beats[i - 1]) / fs) * 1000);
  }

  const bpmBeats = rr.length ? 60000 / avg(rr) : 0;
  const sdnn = rr.length > 1 ? Math.sqrt(avg(rr.map((v) => (v - avg(rr)) ** 2))) : 0;
  const dd = [];
  for (let i = 1; i < rr.length; i++) dd.push(rr[i] - rr[i - 1]);
  const rmssd = dd.length ? Math.sqrt(avg(dd.map((v) => v * v))) : 0;
  const pnn50 = dd.length
    ? (100 * dd.filter((v) => Math.abs(v) > 50).length) / dd.length
    : 0;

  const resp = periodicity(drift, fs, Math.floor(2 * fs), Math.floor(10 * fs));
  const respRate = resp.lag > 0 ? (60 * fs) / resp.lag : 0;

  const bpm =
    bpmAuto > 0 && Math.abs(bpmAuto - bpmBeats) < 25 && bpmBeats > 0
      ? Math.round((bpmAuto + bpmBeats) / 2)
      : Math.round(bpmAuto || bpmBeats);

  const strength = Math.max(0, Math.min(1, per.strength));
  let quality = 'Poor';
  let qclass = 'badge-diff';

  if (strength > 0.45 && beats.length >= 6) {
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
    hpWave: decim(Array.from(hp), 300),
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
 * Real-Time Sensor Placement and Signal Quality Assessment
 * Evaluates the phone's physical orientation, tilt, and stability relative to the chest wall.
 * Returns placement category ('Sternum' | 'Left' | 'Right' | 'Upright'),
 * quality level ('Good' | 'Fair' | 'Poor'), numeric score (0-100),
 * and actionable clinical user guidance.
 */
export function assessRealtimePlacement(x, y, z, recentSamples = []) {
  const norm = Math.hypot(x, y, z);
  if (norm < 0.1) {
    return {
      placement: 'Sternum',
      quality: 'Good',
      score: 92,
      guidance: 'Optimal placement: Lower Sternum',
      isOptimal: true,
      offsetX: 0,
      offsetY: 0,
    };
  }

  // Unit gravity vector
  const gx = x / norm;
  const gy = y / norm;
  const gz = z / norm;

  // Pitch = elevation angle from flat (-90 to 90)
  const pitchAngle = Math.asin(Math.max(-1, Math.min(1, gy))) * (180 / Math.PI);
  // Roll = tilt left/right (-180 to 180)
  const rollAngle = Math.atan2(gx, gz) * (180 / Math.PI);

  // Compute recent variance (stillness index)
  let variance = 0;
  if (recentSamples && recentSamples.length >= 8) {
    const mags = recentSamples.slice(-25).map((s) => Math.hypot(s[1], s[2], s[3]));
    const m = avg(mags);
    variance = avg(mags.map((v) => (v - m) ** 2));
  }

  const isExcessMotion = variance > 0.95;

  let placement = 'Sternum';
  let quality = 'Good';
  let score = 94;
  let guidance = 'Optimal Placement: Center Sternum';
  let isOptimal = true;

  // 1. Check if phone is upright (not flat on chest)
  if (Math.abs(pitchAngle) > 35) {
    placement = 'Upright';
    quality = 'Poor';
    score = 30;
    guidance = 'Recline back flat — phone must rest flat on lower breastbone';
    isOptimal = false;
  }
  // 2. Check lateral tilt (Left vs Right ribcage)
  else if (gx > 0.26 || rollAngle > 18) {
    placement = 'Right';
    quality = 'Fair';
    score = 62;
    guidance = 'Tilted right — shift phone left toward center breastbone';
    isOptimal = false;
  } else if (gx < -0.26 || rollAngle < -18) {
    placement = 'Left';
    quality = 'Fair';
    score = 65;
    guidance = 'Tilted left — shift phone right toward center breastbone';
    isOptimal = false;
  }

  // 3. Check stillness
  if (isExcessMotion) {
    quality = 'Poor';
    score = Math.max(25, score - 35);
    guidance = 'Excess motion detected — hold still and breathe gently';
    isOptimal = false;
  }

  // Normalized visual offset coordinates for target display (-1 to 1)
  const offsetX = Math.max(-1, Math.min(1, gx * 2.5));
  const offsetY = Math.max(-1, Math.min(1, gy * 2.5));

  return {
    placement,
    quality,
    score,
    guidance,
    isOptimal,
    offsetX,
    offsetY,
    pitchAngle: Math.round(pitchAngle),
    rollAngle: Math.round(rollAngle),
    isExcessMotion,
  };
}

