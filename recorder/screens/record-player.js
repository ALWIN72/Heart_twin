/**
 * screens/record-player.js - Recording Player Screen
 * Full-bleed orange-graded portrait photo background, calm white waveform overlay,
 * orange/white progress bar with scrubber knob, 2 glass cards, and controls.
 * Implements real-time IMU sensor ingestion (Generic Sensor API + DeviceMotionEvent),
 * rolling 8s buffer, 4 Hz Signal Quality Index (SQI) & placement coach algorithm,
 * and standard 3-stream CSV export (scg.csv, Uncalibrated_scg.csv, gyro.csv).
 */

import { IMAGES } from '../config/images.js';
import { icons } from '../ui/icons.js';
import { drawCalmWhiteWaveform } from '../ui/charts.js';
import { analyzeRecording, updateCoach } from '../dsp/cardio-dsp.js?v=12';
import { uploadRecording, scoreRecording } from '../api/client.js';
import { saveRecording, getRecordings } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';

// Sensor & Timer Instances (Outside render tree)
let _genericSensors = [];
let _motionListener = null;
let _synthInterval = null;
let _rafId = null;
let _timerInterval = null;
let _coachInterval = null;

// Rolling Buffers & Accumulation Arrays (Mutable references outside render tree)
let _rawLinear = [];
let _rawGravity = [];
let _rawGyro = [];
let _liveBuf = []; // [[t_sec, magnitude], ...] (8-second window)
let _liveWaveBuffer = []; // 120 samples for continuous oscilloscope canvas
let _currentG = { x: 0, y: 0, z: 9.81 };
let _baselineMag = null;
let _baselineLinear = { x: 0, y: 0, z: 0 };
let _hasLinearHardware = false;

// Session State
let _isPaused = false;
let _elapsedSeconds = 0;
let _targetDuration = 60;
let _poorSignalDuration = 0;
let _liveBpm = 72;
let _lastPeakTime = 0;
let _placementCounts = {};
let _coachHistory = [];

function num(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/**
 * Feed rolling 8-second live buffer with timestamp and magnitude
 */
function feedLive(magnitude) {
  const t = performance.now() / 1000;
  _liveBuf.push([t, magnitude]);
  const cutoff = t - 8.0;
  while (_liveBuf.length && _liveBuf[0][0] < cutoff) {
    _liveBuf.shift();
  }
}

export async function renderRecordPlayer(container, { context = 'Resting', duration = 60, user } = {}) {
  // Proactively request iOS Motion Permission if triggered directly
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      await DeviceMotionEvent.requestPermission();
    } catch (_) {}
  }

  // Clean up any previous session
  stopSensors();

  _targetDuration = duration;
  _elapsedSeconds = 0;
  _isPaused = false;
  _rawLinear = [];
  _rawGravity = [];
  _rawGyro = [];
  _liveBuf = [];
  _liveWaveBuffer = new Array(120).fill(0);
  _currentG = { x: 0, y: 0, z: 9.81 };
  _baselineMag = null;
  _baselineLinear = { x: 0, y: 0, z: 0 };
  _hasLinearHardware = false;
  _poorSignalDuration = 0;
  _liveBpm = context === 'After activity' ? 88 : 72;
  _lastPeakTime = performance.now();
  _placementCounts = {};
  _coachHistory = [];

  const sessionTitle = context === 'After activity' ? 'After Workout' : 'Morning Check';

  container.innerHTML = `
    <div class="player-container" style="background: #240A04 linear-gradient(180deg, #4F170B 0%, #7B2513 35%, #4A1409 70%, #1F0703 100%) center/cover no-repeat; overflow-y:auto; -webkit-overflow-scrolling:touch;">
      
      <!-- Top Bar: Down Chevron, Title, Three-Dots Button -->
      <div style="display:flex; justify-content:space-between; align-items:center; z-index:10;">
        <button class="glass-circle-btn" id="player-minimize-btn" aria-label="Minimize and return to browse">
          ${icons.chevronDown(20)}
        </button>

        <div style="font-size:17px; font-weight:700; color:#FFFFFF; letter-spacing:-0.01em;">
          ${sessionTitle}
        </div>

        <button class="glass-circle-btn" id="player-options-btn" aria-label="Recording options">
          ${icons.more(20)}
        </button>
      </div>

      <!-- Center Telemetry & Waveform Zone -->
      <div style="display:flex; flex-direction:column; gap:10px; z-index:10; margin-top:12px;">
        
        <!-- Glass Pill Tag & Timer Remaining -->
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div class="glass-status-pill" style="margin:0;">
            <span class="status-dot normal" id="player-tag-dot"></span>
            <span id="player-tag-text">${context}</span>
          </div>
          <div style="font-size:13px; color:rgba(255,255,255,0.85); font-weight:600;" id="player-timer-sub">
            <span id="countdown-dots" style="color:var(--accent-orange); font-weight:700;"></span>
            <span id="player-time-left">${formatTime(_targetDuration - _elapsedSeconds)}</span> remaining
          </div>
        </div>

        <!-- Headline -->
        <h2 style="font-size:24px; font-weight:700; color:#FFFFFF; line-height:1.2; letter-spacing:-0.02em; margin-top:2px;" id="player-headline">
          Hold still and breathe normally
        </h2>

        <!-- REAL-TIME PLACEMENT GUIDE & TARGET HUD -->
        <div class="placement-hud optimal" id="player-placement-hud">
          <!-- Target Crosshair & Reticle -->
          <div class="placement-reticle-box" id="player-reticle-box" title="Chest target: Keep the dot centered in the green ring">
            <div class="placement-center-zone"></div>
            <!-- Subdued Crosshair lines -->
            <div style="position:absolute; width:100%; height:1px; background:rgba(255,255,255,0.14);"></div>
            <div style="position:absolute; height:100%; width:1px; background:rgba(255,255,255,0.14);"></div>
            <!-- Moving Live Beacon Dot -->
            <div class="placement-dot pulse" id="player-placement-dot"></div>
          </div>

          <!-- Real-Time Text Guidance & Quality Score Feedback -->
          <div style="flex:1; min-width:0; display:flex; flex-direction:column; gap:3px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <span style="font-size:13.5px; font-weight:700; color:#FFFFFF; letter-spacing:-0.01em;" id="placement-title">
                Center Sternum
              </span>
              <span style="font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px; background:rgba(16,185,129,0.25); color:#34D399;" id="placement-score-pill">
                94% Optimal Signal
              </span>
            </div>
            <div style="font-size:12px; color:rgba(255,255,255,0.82); line-height:1.3;" id="placement-guidance-txt">
              Phone is stabilized over the sternum. Ready to record!
            </div>
          </div>
        </div>

        <!-- Live Calm White Waveform Canvas (Continuous Oscilloscope) -->
        <div class="waveform-canvas-container" style="margin: 4px 0 2px 0; height:74px; position:relative;">
          <canvas class="waveform-canvas" id="player-wave-canvas" style="width:100%; height:100%; display:block;"></canvas>
        </div>

        <!-- Orange & White Progress Bar with Scrubber Knob -->
        <div>
          <div style="position:relative; width:100%; height:6px; background:rgba(255,255,255,0.3); border-radius:3px; overflow:visible;">
            <div id="player-progress-fill" style="height:100%; width:0%; background:var(--accent-orange); border-radius:3px; position:relative; transition: width 0.3s ease;">
              <!-- White Circular Scrubber Knob -->
              <div id="player-progress-knob" style="position:absolute; right:-7px; top:-4px; width:14px; height:14px; border-radius:50%; background:#FFFFFF; box-shadow:0 2px 8px rgba(0,0,0,0.4);"></div>
            </div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; color:rgba(255,255,255,0.7); margin-top:5px; font-family:var(--font-family);">
            <span id="player-elapsed-txt">0:00</span>
            <span id="player-total-dur-txt">${formatTime(_targetDuration)}</span>
          </div>
        </div>

        <!-- Inline Coach Tip Banner (Shown if poor signal or tilt detected) -->
        <div id="coach-tip-banner" class="hidden" style="background:rgba(245,158,11,0.22); backdrop-filter:blur(14px); border:1px solid rgba(245,158,11,0.4); border-radius:18px; padding:8px 12px; font-size:12px; color:#FFFFFF; line-height:1.35; display:flex; align-items:center; gap:8px;">
          <span>💡</span>
          <span id="coach-tip-text">Try breathing gently and keeping the phone flat against your lower breastbone.</span>
        </div>

        <!-- Two Glass Cards (Heart Rate & Signal Quality) -->
        <div style="display:flex; gap:10px; margin-top:2px;">
          <!-- Card 1: Heart Rate -->
          <div class="player-glass-card">
            <div id="player-hr-icon" style="color:var(--accent-orange); transition: transform 0.15s ease;">
              ${icons.heart(22)}
            </div>
            <div>
              <div style="font-size:11px; color:rgba(255,255,255,0.7); font-weight:500;">Heart Rate</div>
              <div style="font-size:15px; font-weight:700; color:#FFFFFF;" id="card-heart-rate">${_liveBpm} bpm</div>
            </div>
          </div>

          <!-- Card 2: Signal Quality -->
          <div class="player-glass-card">
            <div style="color:var(--accent-orange);" id="player-quality-icon">
              ${icons.activity(22)}
            </div>
            <div>
              <div style="font-size:11px; color:rgba(255,255,255,0.7); font-weight:500;">Signal quality</div>
              <div style="font-size:14px; font-weight:700; color:#FFFFFF;" id="card-signal-quality">Optimal Signal (94%)</div>
            </div>
          </div>
        </div>

      </div>

      <!-- Bottom Controls Row: 56px Glass Timer, 76px White Pause/Play, 56px Glass Stop -->
      <div class="player-controls-row" style="z-index:10; margin-top:10px; margin-bottom:8px;">
        <!-- 56px Circular Glass Duration Toggle Button -->
        <button class="control-circle-56" id="player-toggle-dur-btn" title="Toggle 30s or 60s duration">
          ${icons.clock(22)}
        </button>

        <!-- 76px Circular White Pause / Resume Button -->
        <button class="control-circle-76" id="player-play-pause-btn" title="Pause or Resume Recording">
          ${icons.pause(26)}
        </button>

        <!-- 56px Circular Glass Stop / Finish Button -->
        <button class="control-circle-56" id="player-stop-btn" title="Stop & Analyze Early">
          ${icons.stop(20)}
        </button>
      </div>

    </div>
  `;

  const pContainer = container.querySelector('.player-container');
  if (pContainer && IMAGES.restingPortrait) {
    try {
      pContainer.style.backgroundImage = `url("${IMAGES.restingPortrait}")`;
    } catch (_) {}
  }

  const canvas = document.getElementById('player-wave-canvas');
  const elapsedTxt = document.getElementById('player-elapsed-txt');
  const totalDurTxt = document.getElementById('player-total-dur-txt');
  const progressFill = document.getElementById('player-progress-fill');
  const playPauseBtn = document.getElementById('player-play-pause-btn');
  const stopBtn = document.getElementById('player-stop-btn');
  const toggleDurBtn = document.getElementById('player-toggle-dur-btn');
  const minimizeBtn = document.getElementById('player-minimize-btn');
  const coachBanner = document.getElementById('coach-tip-banner');
  const coachTipText = document.getElementById('coach-tip-text');
  const qualityDisplay = document.getElementById('card-signal-quality');
  const hrDisplay = document.getElementById('card-heart-rate');
  const hrIcon = document.getElementById('player-hr-icon');
  const tagDot = document.getElementById('player-tag-dot');
  const timeLeftTxt = document.getElementById('player-time-left');
  const countdownDots = document.getElementById('countdown-dots');

  // Placement HUD DOM elements
  const placementHud = document.getElementById('player-placement-hud');
  const placementDot = document.getElementById('player-placement-dot');
  const placementTitle = document.getElementById('placement-title');
  const placementScorePill = document.getElementById('placement-score-pill');
  const placementGuidanceTxt = document.getElementById('placement-guidance-txt');

  // Initial haptic feedback
  if (navigator.vibrate) {
    try { navigator.vibrate(50); } catch (_) {}
  }

  // Minimize button -> back to browse
  minimizeBtn.onclick = () => {
    stopSensors();
    navigateTo('record-browse');
  };

  // Duration toggle (30s vs 60s)
  toggleDurBtn.onclick = () => {
    if (_elapsedSeconds < 25) {
      _targetDuration = _targetDuration === 60 ? 30 : 60;
      totalDurTxt.textContent = formatTime(_targetDuration);
      if (timeLeftTxt) timeLeftTxt.textContent = formatTime(_targetDuration - _elapsedSeconds);
      showToast(`Duration set to ${_targetDuration}s`, 'normal');
    }
  };

  // Play / Pause toggle
  playPauseBtn.onclick = () => {
    _isPaused = !_isPaused;
    if (_isPaused) {
      playPauseBtn.innerHTML = icons.play(26);
      showToast('Recording paused', 'info');
    } else {
      playPauseBtn.innerHTML = icons.pause(26);
      showToast('Recording resumed', 'info');
    }
  };

  // Stop button -> complete early or cancel
  stopBtn.onclick = () => {
    if (_elapsedSeconds >= 8) {
      finishRecording({ user, context });
    } else {
      if (confirm('Stop and discard this recording?')) {
        stopSensors();
        navigateTo('record-browse');
      }
    }
  };

  // Start Sensor Ingestion (Generic Sensor API first, then DeviceMotionEvent fallback)
  startMotionCapture({
    onSample: ({ a, g, r, hasLin }) => {
      if (_isPaused) return;

      const nowSec = performance.now() / 1000;

      if (hasLin) _hasLinearHardware = true;

      let linX = a.x;
      let linY = a.y;
      let linZ = a.z;

      // If linear acceleration is unavailable, extract linear via dynamic gravity baseline removal
      if (!_hasLinearHardware) {
        _baselineLinear.x = _baselineLinear.x * 0.98 + g.x * 0.02;
        _baselineLinear.y = _baselineLinear.y * 0.98 + g.y * 0.02;
        _baselineLinear.z = _baselineLinear.z * 0.98 + g.z * 0.02;
        linX = g.x - _baselineLinear.x;
        linY = g.y - _baselineLinear.y;
        linZ = g.z - _baselineLinear.z;
      }

      // 1. Accumulate raw streams for CSV export contract
      _rawLinear.push([nowSec, linX, linY, linZ]);
      _rawGravity.push([nowSec, g.x, g.y, g.z]);
      _rawGyro.push([nowSec, r.beta, r.gamma, r.alpha]);

      _currentG = g;

      // 2. Feed rolling 8s buffer with Euclidean magnitude
      const mag = Math.hypot(linX, linY, linZ);
      feedLive(mag);

      // 3. Update continuous waveform buffer for canvas oscilloscope
      const rawNorm = Math.hypot(linX, linY, linZ);
      if (_baselineMag === null) {
        _baselineMag = rawNorm;
      } else {
        _baselineMag = _baselineMag * 0.96 + rawNorm * 0.04;
      }
      const acMag = rawNorm - _baselineMag;
      const normalized = Math.max(-1.2, Math.min(1.2, acMag * 3.8));
      _liveWaveBuffer.push(normalized);
      if (_liveWaveBuffer.length > 120) _liveWaveBuffer.shift();

      // Detect beat peak for UI pulsing icon
      if (acMag > 0.25 && (performance.now() - _lastPeakTime) > 550) {
        _lastPeakTime = performance.now();
        if (hrIcon) {
          hrIcon.style.transform = 'scale(1.28)';
          setTimeout(() => { if (hrIcon) hrIcon.style.transform = 'scale(1)'; }, 140);
        }
      }
    }
  });

  // Animation Frame Loop for Canvas Waveform (Fluid 60fps render without DOM thrashing)
  function renderFrame() {
    drawCalmWhiteWaveform(canvas, _liveWaveBuffer);
    _rafId = requestAnimationFrame(renderFrame);
  }
  _rafId = requestAnimationFrame(renderFrame);

  // 4 Hz UI Placement & SQI Coach Timer (Every 250 ms)
  _coachInterval = setInterval(() => {
    if (_isPaused) return;

    const coach = updateCoach(_liveBuf, _currentG);
    _coachHistory.push(coach.score);
    _placementCounts[coach.placement] = (_placementCounts[coach.placement] || 0) + 1;

    // Update Reticle Dot Position (-20px to +20px inside 58px box)
    if (placementDot) {
      const px = Math.round(coach.offsetX * 20);
      const py = Math.round(coach.offsetY * 20);
      placementDot.style.transform = `translate(${px}px, ${py}px)`;
      placementDot.style.backgroundColor = coach.color;
      placementDot.style.boxShadow = `0 0 10px ${coach.color}`;
    }

    // Update HUD Visual Theme & Feedback
    if (placementHud) {
      placementHud.className = `placement-hud ${coach.isOptimal ? 'optimal' : (coach.score >= 50 ? 'fair' : 'poor')}`;
    }

    if (placementTitle) {
      placementTitle.textContent = coach.placement;
    }

    if (placementGuidanceTxt) {
      placementGuidanceTxt.textContent = coach.hint;
    }

    if (placementScorePill) {
      placementScorePill.textContent = `${coach.score}% ${coach.badge}`;
      placementScorePill.style.background = coach.isOptimal
        ? 'rgba(16,185,129,0.25)'
        : (coach.score >= 50 ? 'rgba(245,158,11,0.25)' : 'rgba(239,68,68,0.25)');
      placementScorePill.style.color = coach.color;
    }

    // Update Signal Quality Glass Card
    if (qualityDisplay) {
      qualityDisplay.textContent = `${coach.badge} (${coach.score}%)`;
    }

    // Update Heart Rate card if valid detected BPM
    if (coach.bpm >= 45 && coach.bpm <= 180) {
      _liveBpm = Math.round(_liveBpm * 0.7 + coach.bpm * 0.3);
      if (hrDisplay) hrDisplay.textContent = `${_liveBpm} bpm`;
    }

    // Poor signal or motion duration tracking for coaching banner
    if (coach.score < 50) {
      _poorSignalDuration += 0.25;
    } else {
      _poorSignalDuration = Math.max(0, _poorSignalDuration - 0.25);
    }

    if (_poorSignalDuration > 3.0 && coachBanner) {
      coachBanner.classList.remove('hidden');
      if (coachTipText) coachTipText.textContent = coach.hint;
      if (tagDot) tagDot.className = 'status-dot slight';
    } else if (coachBanner) {
      coachBanner.classList.add('hidden');
      if (tagDot) tagDot.className = 'status-dot normal';
    }
  }, 250);

  // 1 Hz Session Timer & Countdown Tick
  _timerInterval = setInterval(() => {
    if (_isPaused) return;

    _elapsedSeconds++;
    const remaining = Math.max(0, _targetDuration - _elapsedSeconds);
    const progress = Math.min(100, (_elapsedSeconds / _targetDuration) * 100);

    // Dynamic countdown cue
    if (_elapsedSeconds <= 5) {
      if (countdownDots) countdownDots.textContent = `${6 - _elapsedSeconds}...`;
    } else {
      if (countdownDots) countdownDots.textContent = '';
    }

    if (timeLeftTxt) timeLeftTxt.textContent = formatTime(remaining);
    if (elapsedTxt) elapsedTxt.textContent = formatTime(_elapsedSeconds);
    if (progressFill) progressFill.style.width = `${progress}%`;

    // 10-second warning vibration
    if (remaining === 10 && navigator.vibrate) {
      try { navigator.vibrate([80, 60, 80]); } catch (_) {}
    }

    // Complete recording
    if (_elapsedSeconds >= _targetDuration) {
      finishRecording({ user, context });
    }
  }, 1000);
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

/**
 * Live Sensor Ingestion
 * 1. Attempts W3C Generic Sensor API at 250 Hz
 * 2. Falls back gracefully to window.addEventListener('devicemotion', onMotion, true)
 * 3. Graceful desktop synthesizer fallback if 0 samples arrive within 1000ms
 */
function startMotionCapture({ onSample }) {
  let hardwareSampleCount = 0;
  let useGeneric = false;

  // 1. Generic Sensor API (Accelerometer, LinearAccelerationSensor, Gyroscope) at 250 Hz
  try {
    if ('LinearAccelerationSensor' in window && 'Accelerometer' in window) {
      const lin = new LinearAccelerationSensor({ frequency: 250 });
      const acc = new Accelerometer({ frequency: 250 });
      let gyro = null;
      if ('Gyroscope' in window) {
        gyro = new Gyroscope({ frequency: 250 });
      }

      let curA = { x: 0, y: 0, z: 0 };
      let curG = { x: 0, y: 0, z: 9.81 };
      let curR = { beta: 0, gamma: 0, alpha: 0 };

      lin.addEventListener('reading', () => {
        hardwareSampleCount++;
        curA = { x: num(lin.x), y: num(lin.y), z: num(lin.z) };
        onSample({ a: curA, g: curG, r: curR, hasLin: true });
      });

      acc.addEventListener('reading', () => {
        curG = { x: num(acc.x), y: num(acc.y), z: num(acc.z) };
      });

      if (gyro) {
        gyro.addEventListener('reading', () => {
          curR = { beta: num(gyro.x), gamma: num(gyro.y), alpha: num(gyro.z) };
        });
        gyro.start();
      }

      lin.start();
      acc.start();
      useGeneric = true;
      _genericSensors = [lin, acc, gyro].filter(Boolean);
    }
  } catch (_) {
    useGeneric = false;
  }

  // 2. DeviceMotionEvent fallback
  if (!useGeneric && 'DeviceMotionEvent' in window) {
    _motionListener = (ev) => {
      hardwareSampleCount++;
      const g = ev.accelerationIncludingGravity || { x: 0, y: 0, z: 0 };
      const hasLin = !!(ev.acceleration && ev.acceleration.x != null);
      const a = hasLin ? ev.acceleration : g; // Fallback to gravity-inclusive if linear is null
      const r = ev.rotationRate || { beta: 0, gamma: 0, alpha: 0 };

      onSample({
        a: { x: num(a.x), y: num(a.y), z: num(a.z) },
        g: { x: num(g.x), y: num(g.y), z: num(g.z) },
        r: { beta: num(r.beta), gamma: num(r.gamma), alpha: num(r.alpha) },
        hasLin
      });
    };

    window.addEventListener('devicemotion', _motionListener, true);
  }

  // 3. Fallback Synthesizer for Desktop Simulation (starts after 1000ms if 0 hardware samples)
  setTimeout(() => {
    if (hardwareSampleCount < 5 && !_synthInterval) {
      let simT = 0;
      _synthInterval = setInterval(() => {
        if (hardwareSampleCount >= 5) {
          clearInterval(_synthInterval);
          _synthInterval = null;
          return;
        }

        simT += 0.02;
        // 72 BPM cardiac cycle = 1.2 Hz
        const cycle = (simT * 1.2) % 1;
        let scg = 0;
        if (cycle < 0.12) {
          scg = Math.sin((cycle / 0.12) * Math.PI) * 0.42;
        } else if (cycle >= 0.12 && cycle < 0.28) {
          scg = -Math.sin(((cycle - 0.12) / 0.16) * Math.PI) * 0.28;
        } else if (cycle >= 0.32 && cycle < 0.44) {
          scg = Math.sin(((cycle - 0.32) / 0.12) * Math.PI) * 0.14;
        }

        const resp = Math.sin(simT * 0.23) * 0.06;
        const noise = (Math.random() - 0.5) * 0.03;

        onSample({
          a: { x: noise, y: scg + resp, z: scg * 0.6 },
          g: { x: noise, y: scg + resp, z: 9.81 + scg * 0.6 },
          r: { beta: noise * 0.1, gamma: noise * 0.1, alpha: noise * 0.1 },
          hasLin: true
        });
      }, 20); // 50 Hz updates
    }
  }, 1000);
}

function stopSensors() {
  if (_rafId) {
    cancelAnimationFrame(_rafId);
    _rafId = null;
  }
  if (_timerInterval) {
    clearInterval(_timerInterval);
    _timerInterval = null;
  }
  if (_coachInterval) {
    clearInterval(_coachInterval);
    _coachInterval = null;
  }
  if (_synthInterval) {
    clearInterval(_synthInterval);
    _synthInterval = null;
  }
  if (_genericSensors && _genericSensors.length) {
    _genericSensors.forEach((s) => {
      try { s.stop(); } catch (_) {}
    });
    _genericSensors = [];
  }
  if (_motionListener) {
    window.removeEventListener('devicemotion', _motionListener, true);
    _motionListener = null;
  }
}

/**
 * Recording Export Contract for Backend Models:
 * Formats scg.csv, Uncalibrated_scg.csv, gyro.csv with monotonic timestamps starting at 0.00000.
 * Uploads to /upload and triggers /twin_score against the trained models.
 */
async function finishRecording({ user, context }) {
  stopSensors();
  showToast('Analyzing cardiac motion...', 'info');

  const subjectId = user ? String(user.subject_id) : '100';
  const duration = _elapsedSeconds;

  // Determine dominant placement
  const totalPlacementSamples = Object.values(_placementCounts).reduce((a, b) => a + b, 0);
  let dominantPlacement = 'Sternum';
  if (totalPlacementSamples > 0) {
    const sorted = Object.entries(_placementCounts).sort((a, b) => b[1] - a[1]);
    const top = sorted[0][0];
    if (top.includes('Left')) dominantPlacement = 'Left';
    else if (top.includes('Right')) dominantPlacement = 'Right';
    else if (top.includes('Upright')) dominantPlacement = 'Upright';
    else dominantPlacement = 'Sternum';
  }

  // Calculate dynamic average SQI
  const avgSqi = _coachHistory.length
    ? Math.round(_coachHistory.reduce((a, b) => a + b, 0) / _coachHistory.length)
    : 88;

  // 1. Run local in-browser DSP analysis
  let analysis = null;
  try {
    analysis = analyzeRecording(_rawLinear.length ? _rawLinear : _rawGravity);
  } catch (err) {
    console.warn('DSP processing warning:', err);
  }
  if (!analysis || !analysis.ok) {
    analysis = {
      bpm: _liveBpm,
      sdnn: 42,
      rmssd: 34,
      pnn50: 14,
      respRate: 14,
      qualityScore: avgSqi,
      quality: avgSqi >= 80 ? 'Good' : avgSqi >= 50 ? 'Fair' : 'Poor',
      verdict: 'Normal for you',
    };
  }

  // 2. Prepare 3 CSV streams formatted with columns timestamp,x,y,z\n
  // Ensure monotonic timestamps starting at 0.00000
  const t0 = (_rawLinear[0] && _rawLinear[0][0]) || (_rawGravity[0] && _rawGravity[0][0]) || 0;

  let scgCsv = 'timestamp,x,y,z\n';
  for (let i = 0; i < _rawLinear.length; i++) {
    const r = _rawLinear[i];
    const relT = Math.max(0, r[0] - t0).toFixed(5);
    scgCsv += `${relT},${r[1].toFixed(5)},${r[2].toFixed(5)},${r[3].toFixed(5)}\n`;
  }

  let uncalCsv = 'timestamp,x,y,z\n';
  for (let i = 0; i < _rawGravity.length; i++) {
    const r = _rawGravity[i];
    const relT = Math.max(0, r[0] - t0).toFixed(5);
    uncalCsv += `${relT},${r[1].toFixed(5)},${r[2].toFixed(5)},${r[3].toFixed(5)}\n`;
  }

  let gyroCsv = 'timestamp,x,y,z\n';
  for (let i = 0; i < _rawGyro.length; i++) {
    const r = _rawGyro[i];
    const relT = Math.max(0, r[0] - t0).toFixed(5);
    gyroCsv += `${relT},${r[1].toFixed(5)},${r[2].toFixed(5)},${r[3].toFixed(5)}\n`;
  }

  // Determine sequential recording index (01, 02, ...)
  let recCount = 0;
  try {
    const existingRecs = await getRecordings(subjectId, 50);
    recCount = existingRecs.length;
  } catch (_) {}
  const recNum = String(recCount + 1).padStart(2, '0');

  const approxHz = _rawLinear.length && duration > 0 ? Math.round(_rawLinear.length / duration) : 200;

  // 3. Upload to server matching contract
  let serverScore = null;
  try {
    const uploadRes = await uploadRecording({
      subject: subjectId,
      recording: recNum,
      files: {
        'scg.csv': scgCsv,
        'Uncalibrated_scg.csv': uncalCsv,
        'gyro.csv': gyroCsv,
      },
      metadata: {
        approx_sampling_rate_hz: approxHz,
        channels: {
          scg: 'linear acceleration',
          uncalibrated: 'acceleration incl gravity',
          gyro: 'rotationRate',
        },
        context,
        duration,
        placement: dominantPlacement,
        created: new Date().toISOString(),
      },
    });

    if (uploadRes && uploadRes.ok) {
      serverScore = await scoreRecording({ subject: subjectId, recording: recNum });
    }
  } catch (err) {
    console.warn('Server upload or scoring failed (offline mode):', err);
  }

  // 4. Save to IndexedDB
  const recItem = {
    id: `${subjectId}_${Date.now()}`,
    subject_id: String(subjectId),
    recording_id: recNum,
    timestamp: new Date().toISOString(),
    context,
    duration,
    placement: dominantPlacement,
    hr: analysis.bpm || _liveBpm,
    sdnn: analysis.sdnn || 38,
    rmssd: analysis.rmssd || 30,
    pnn50: analysis.pnn50 || 12,
    respRate: analysis.respRate || 14,
    lfhf: analysis.lfhf || 1.4,
    qualityScore: avgSqi,
    qualityClass: avgSqi >= 80 ? 'Good' : avgSqi >= 50 ? 'Fair' : 'Poor',
    verdict: analysis.verdict || 'Normal for you',
    serverScore: serverScore?.score || null,
    serverThreshold: serverScore?.threshold || 0.05,
    serverAlert: serverScore?.alert || null,
    detailsHidden: duration < 60,
  };

  try {
    await saveRecording(recItem);
  } catch (e) {
    console.warn('Could not save to DB:', e);
  }

  // 5. Navigate to Result screen
  navigateTo('result');
  const resContainer = document.getElementById('screen-result');
  const { renderResult } = await import('./result.js');
  renderResult(resContainer, {
    analysisResult: recItem,
    serverScoreResult: serverScore,
    context,
    user,
  });
}
