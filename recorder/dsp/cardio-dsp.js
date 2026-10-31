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
 * Normalized Autocorrelation Periodicity
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
  for (let lag = minLag; lag <= maxLag && lag < len; lag++) {
    let s = 0;
    for (let i = 0; i + lag < len; i++) {
      s += centered[i] * centered[i + lag];
    }
    const normCorr = s / ac0;
    if (normCorr > maxCorr) {
      maxCorr = normCorr;
      bestLag = lag;
    }
  }
  return { strength: Math.max(0, maxCorr), lag: bestLag };
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
 * Real-Time Signal Quality Index (SQI) & Placement Coach Algorithm
 * Evaluates live 8-second rolling magnitude buffer every 250 ms (4 Hz).
 * Combines dynamic sampling frequency, jerk motion artifact, energy envelope,
 * autocorrelation periodicity in 40-180 BPM range, and gravity-vector orientation.
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
      placement: 'Sternum',
      offsetX: 0,
      offsetY: 0,
      isOptimal: false,
    };
  }

  const mags = liveBuf.map((p) => p[1]);
  const ts = liveBuf.map((p) => p[0]);
  const dur = Math.max(0.001, ts[ts.length - 1] - ts[0]);
  const fs = mags.length / dur; // Dynamic true sampling rate

  // 1. Mean-centering & standard deviation
  const m = mags.reduce((a, b) => a + b, 0) / mags.length;
  const x = mags.map((v) => v - m);
  const sd = Math.sqrt(x.reduce((acc, v) => acc + v * v, 0) / x.length);

  // 2. Jerk / Motion artifact calculation
  let jerk = 0;
  for (let i = 1; i < x.length; i++) jerk += Math.abs(x[i] - x[i - 1]);
  jerk /= Math.max(1, x.length - 1);

  // 3. Squared Energy Envelope (80ms moving average)
  const sq = x.map((v) => v * v);
  const wSize = Math.max(2, Math.round(fs * 0.08));
  const env = movingAverageCentered(sq, wSize);

  // 4. Normalized Autocorrelation in physiological lag range (40 - 180 BPM)
  const minLag = Math.floor(0.33 * fs); // 180 BPM
  const maxLag = Math.floor(1.50 * fs); // 40 BPM
  const per = calculatePeriodicity(env, minLag, maxLag);
  const strength = per.strength; // [0.0, 1.0]

  // Orientation analysis from gravity vector
  const normG = Math.hypot(g.x, g.y, g.z) || 9.81;
  const gx = g.x / normG;
  const gy = g.y / normG;
  const gz = g.z / normG;

  const pitchAngle = Math.asin(Math.max(-1, Math.min(1, gy))) * (180 / Math.PI);
  const rollAngle = Math.atan2(gx, Math.abs(gz) > 0.05 ? gz : 0.05) * (180 / Math.PI);
  const offsetX = Math.max(-1, Math.min(1, gx * 3.2));
  const offsetY = Math.max(-1, Math.min(1, gy * 3.2));

  let placement = 'Center Sternum';
  let isTiltOptimal = true;

  if (Math.abs(pitchAngle) > 35) {
    placement = 'Phone Upright';
    isTiltOptimal = false;
  } else if (gx > 0.25 || rollAngle > 18) {
    placement = 'Tilted Right';
    isTiltOptimal = false;
  } else if (gx < -0.25 || rollAngle < -18) {
    placement = 'Tilted Left (Apex)';
    isTiltOptimal = false;
  }

  // 5. Decision Tree for Guidance & SQI
  let score, badge, message, hint, color;
  if (sd < 0.003) {
    score = 12;
    badge = 'No Signal';
    message = 'No heartbeat detected';
    hint = 'Place phone firmly on the center of your chest.';
    color = '#EF4444';
  } else if (!isTiltOptimal && placement === 'Phone Upright') {
    score = 30;
    badge = 'Phone Upright';
    message = 'Recline flat';
    hint = 'Recline back flat — phone must rest flat on lower breastbone.';
    color = '#EF4444';
  } else if (!isTiltOptimal && placement.includes('Left')) {
    score = 55;
    badge = 'Tilted Left';
    message = 'Shift toward center';
    hint = 'Tilted left toward ribcage — shift phone right toward center breastbone.';
    color = '#F59E0B';
  } else if (!isTiltOptimal && placement.includes('Right')) {
    score = 55;
    badge = 'Tilted Right';
    message = 'Shift toward center';
    hint = 'Tilted right — shift phone left toward center breastbone.';
    color = '#F59E0B';
  } else if (jerk > sd * 0.9) {
    score = 35;
    badge = 'Motion Detected';
    message = 'Excessive movement';
    hint = 'Relax shoulders and breathe gently.';
    color = '#EF4444';
  } else if (strength > 0.42) {
    score = 94;
    badge = 'Optimal Signal';
    message = 'Clear heartbeat detected';
    hint = 'Phone is stabilized over the sternum. Ready to record!';
    color = '#10B981';
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

  const isOptimal = score >= 80;
  const bpm = per.lag > 0 ? Math.round((60 * fs) / per.lag) : 0;

  return {
    score,
    badge,
    message,
    hint,
    color,
    fs,
    bpm,
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

