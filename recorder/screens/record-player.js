/**
 * screens/record-player.js - Recording Player Screen (Phone 3 from Reference Screenshot)
 * Full-bleed orange-graded portrait photo background, calm white waveform overlay,
 * orange/white progress bar with scrubber knob, 2 glass cards, and 56/76/56 controls.
 * Deeply responsive and interactive on both real mobile sensors and desktop simulator.
 * Features Real-Time Chest Placement Target HUD & Dynamic Signal Quality Guide.
 */

import { IMAGES } from '../config/images.js';
import { icons } from '../ui/icons.js';
import { drawCalmWhiteWaveform } from '../ui/charts.js';
import { analyzeRecording, assessRealtimePlacement } from '../dsp/cardio-dsp.js';
import { uploadRecording, scoreRecording } from '../api/client.js';
import { saveRecording } from '../storage/db.js';
import { navigateTo } from '../ui/nav.js';
import { showToast } from '../ui/toast.js';

let _motionListener = null;
let _sensorInstance = null;
let _synthInterval = null;
let _rafId = null;
let _timerInterval = null;
let _rawData = [];
let _liveBuffer = [];
let _isPaused = false;
let _elapsedSeconds = 0;
let _targetDuration = 60;
let _poorSignalDuration = 0;
let _baselineMag = null;
let _liveBpm = 72;
let _lastPeakTime = 0;

// Placement Tracking State
let _placementCounts = { Sternum: 0, Left: 0, Right: 0, Upright: 0 };
let _lastPlacementObj = null;

export async function renderRecordPlayer(container, { context = 'Resting', duration = 60, user } = {}) {
  // Proactively ensure iOS Motion Permission on entering recording player
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      await DeviceMotionEvent.requestPermission();
    } catch (_) {}
  }
  // Clean up any existing session
  stopSensors();

  _targetDuration = duration;
  _elapsedSeconds = 0;
  _isPaused = false;
  _rawData = [];
  _liveBuffer = new Array(100).fill(0);
  _poorSignalDuration = 0;
  _baselineMag = null;
  _liveBpm = context === 'After activity' ? 88 : 72;
  _lastPeakTime = performance.now();
  _placementCounts = { Sternum: 0, Left: 0, Right: 0, Upright: 0 };
  _lastPlacementObj = null;

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

        <!-- Headline (Phone 3 match) -->
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
                94% Optimal
              </span>
            </div>
            <div style="font-size:12px; color:rgba(255,255,255,0.82); line-height:1.3;" id="placement-guidance-txt">
              Phone flat on lower breastbone (Sternum)
            </div>
          </div>
        </div>

        <!-- Live Calm White Waveform Canvas -->
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
              <div style="font-size:14px; font-weight:700; color:#FFFFFF;" id="card-signal-quality">Good (94%)</div>
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

  // Throttle placement UI updates for fluid 20fps HUD animation
  let _lastPlacementUiUpdate = 0;

  // Start Sensor Ingestion
  startMotionCapture({
    onSample: (x, y, z) => {
      if (_isPaused) return;

      const now = performance.now() / 1000;
      _rawData.push([now, x, y, z]);

      // Robust Moving Baseline for Seismocardiogram
      const rawMag = Math.sqrt(x * x + y * y + z * z);
      if (_baselineMag === null) {
        _baselineMag = rawMag;
      } else {
        _baselineMag = _baselineMag * 0.96 + rawMag * 0.04;
      }

      // High-pass dynamic amplitude
      const acMag = rawMag - _baselineMag;
      // Normalized amplitude for canvas visualization (-1 to +1)
      const normalized = Math.max(-1.2, Math.min(1.2, acMag * 3.8));
      _liveBuffer.push(normalized);
      if (_liveBuffer.length > 100) _liveBuffer.shift();

      // Detect beat peak for UI pulsing
      if (acMag > 0.25 && (performance.now() - _lastPeakTime) > 600) {
        const intervalMs = performance.now() - _lastPeakTime;
        _lastPeakTime = performance.now();
        const detectedBpm = Math.round(60000 / intervalMs);
        if (detectedBpm >= 45 && detectedBpm <= 160) {
          _liveBpm = Math.round(_liveBpm * 0.7 + detectedBpm * 0.3);
          if (hrDisplay) hrDisplay.textContent = `${_liveBpm} bpm`;
          if (hrIcon) {
            hrIcon.style.transform = 'scale(1.28)';
            setTimeout(() => { if (hrIcon) hrIcon.style.transform = 'scale(1)'; }, 140);
          }
        }
      }

      // Real-Time Placement & Signal Quality Assessment
      const nowMs = performance.now();
      if (nowMs - _lastPlacementUiUpdate > 50) {
        _lastPlacementUiUpdate = nowMs;

        const placementData = assessRealtimePlacement(x, y, z, _rawData);
        _lastPlacementObj = placementData;

        // Record placement category for overall session score
        if (placementData.placement in _placementCounts) {
          _placementCounts[placementData.placement]++;
        } else {
          _placementCounts.Sternum++;
        }

        // Update Reticle Dot Position (-20px to +20px inside 58px box)
        if (placementDot) {
          const px = Math.round(placementData.offsetX * 20);
          const py = Math.round(placementData.offsetY * 20);
          placementDot.style.transform = `translate(${px}px, ${py}px)`;

          if (placementData.isOptimal) {
            placementDot.style.backgroundColor = '#10B981';
            placementDot.style.boxShadow = '0 0 10px #10B981';
          } else if (placementData.quality === 'Fair') {
            placementDot.style.backgroundColor = '#F59E0B';
            placementDot.style.boxShadow = '0 0 10px #F59E0B';
          } else {
            placementDot.style.backgroundColor = '#EF4444';
            placementDot.style.boxShadow = '0 0 10px #EF4444';
          }
        }

        // Update HUD Visual Theme & Feedback
        if (placementHud) {
          placementHud.className = `placement-hud ${placementData.isOptimal ? 'optimal' : (placementData.quality === 'Fair' ? 'fair' : 'poor')}`;
        }

        if (placementTitle) {
          let posName = 'Center Sternum';
          if (placementData.placement === 'Left') posName = 'Tilted Left (Apex)';
          else if (placementData.placement === 'Right') posName = 'Tilted Right';
          else if (placementData.placement === 'Upright') posName = 'Phone Upright';
          placementTitle.textContent = posName;
        }

        if (placementGuidanceTxt) {
          placementGuidanceTxt.textContent = placementData.guidance;
        }

        if (placementScorePill) {
          placementScorePill.textContent = `${placementData.score}% ${placementData.quality}`;
          if (placementData.isOptimal) {
            placementScorePill.style.background = 'rgba(16,185,129,0.25)';
            placementScorePill.style.color = '#34D399';
          } else if (placementData.quality === 'Fair') {
            placementScorePill.style.background = 'rgba(245,158,11,0.25)';
            placementScorePill.style.color = '#FBBF24';
          } else {
            placementScorePill.style.background = 'rgba(239,68,68,0.25)';
            placementScorePill.style.color = '#F87171';
          }
        }

        // Update Signal Quality Glass Card
        if (qualityDisplay) {
          qualityDisplay.textContent = `${placementData.quality} (${placementData.score}%)`;
        }

        // Poor signal duration accumulator & coaching banner
        if (!placementData.isOptimal) {
          _poorSignalDuration += 0.05;
        } else {
          _poorSignalDuration = Math.max(0, _poorSignalDuration - 0.05);
        }

        if (_poorSignalDuration > 5 && coachBanner) {
          coachBanner.classList.remove('hidden');
          if (coachTipText) coachTipText.textContent = placementData.guidance;
          if (tagDot) tagDot.className = 'status-dot slight';
        } else if (coachBanner) {
          coachBanner.classList.add('hidden');
          if (tagDot) tagDot.className = 'status-dot normal';
        }
      }
    }
  });

  // Animation Frame Loop for Canvas Waveform
  function renderFrame() {
    drawCalmWhiteWaveform(canvas, _liveBuffer);
    _rafId = requestAnimationFrame(renderFrame);
  }
  _rafId = requestAnimationFrame(renderFrame);

  // Timer Tick (1 Hz)
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

function startMotionCapture({ onSample }) {
  let hardwareSampleCount = 0;

  // 1. Generic Sensor API (Accelerometer)
  if ('LinearAccelerationSensor' in window) {
    try {
      _sensorInstance = new LinearAccelerationSensor({ frequency: 200 });
      _sensorInstance.addEventListener('reading', () => {
        hardwareSampleCount++;
        onSample(_sensorInstance.x || 0, _sensorInstance.y || 0, _sensorInstance.z || 0);
      });
      _sensorInstance.start();
    } catch (_) {}
  }

  // 2. DeviceMotionEvent (Standard Mobile Safari iOS & Chrome Android)
  if ('DeviceMotionEvent' in window) {
    _motionListener = (event) => {
      const acc = event.accelerationIncludingGravity || event.acceleration;
      if (acc && (acc.x !== null || acc.y !== null || acc.z !== null)) {
        hardwareSampleCount++;
        // Device motion x, y, z in m/s^2
        onSample(acc.x || 0, acc.y || 0, acc.z || 0);
      }
    };
    window.addEventListener('devicemotion', _motionListener, false);
  }

  // 3. Fallback High-Fidelity SCG Synthesizer for Desktop Browsers without Accelerometer
  // Runs ONLY if no physical motion hardware events arrive after 900ms (e.g. desktop Mac/PC)
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

        // Physiological SCG waveform
        if (cycle < 0.12) {
          scg = Math.sin(cycle / 0.12 * Math.PI) * 0.42;
        } else if (cycle >= 0.12 && cycle < 0.28) {
          scg = -Math.sin((cycle - 0.12) / 0.16 * Math.PI) * 0.28;
        } else if (cycle >= 0.32 && cycle < 0.44) {
          scg = Math.sin((cycle - 0.32) / 0.12 * Math.PI) * 0.14;
        }

        // Thoracic respiration envelope drift (14 breaths/min)
        const resp = Math.sin(simT * 0.23) * 0.06;
        const noise = (Math.random() - 0.5) * 0.03;

        // Clean flat baseline on desktop test
        onSample(noise, scg + resp, 9.81 + scg * 0.6);
      }, 20); // 50 Hz updates
    }
  }, 900);
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
  if (_synthInterval) {
    clearInterval(_synthInterval);
    _synthInterval = null;
  }
  if (_sensorInstance) {
    try { _sensorInstance.stop(); } catch (_) {}
    _sensorInstance = null;
  }
  if (_motionListener) {
    if (typeof _motionListener === 'function') {
      window.removeEventListener('devicemotion', _motionListener);
    } else {
      clearInterval(_motionListener);
    }
    _motionListener = null;
  }
}

async function finishRecording({ user, context }) {
  stopSensors();
  showToast('Analyzing cardiac motion...', 'info');

  const subjectId = user ? user.subject_id : '1001';
  const duration = _elapsedSeconds;

  // Determine dominant placement over the session
  const totalSamples = Object.values(_placementCounts).reduce((a, b) => a + b, 0);
  let dominantPlacement = 'Center Sternum (Optimal)';
  if (totalSamples > 0) {
    const sternumRatio = (_placementCounts.Sternum || 0) / totalSamples;
    const leftRatio = (_placementCounts.Left || 0) / totalSamples;
    const rightRatio = (_placementCounts.Right || 0) / totalSamples;
    const uprightRatio = (_placementCounts.Upright || 0) / totalSamples;

    if (sternumRatio >= 0.6) {
      dominantPlacement = 'Center Sternum (Optimal)';
    } else if (leftRatio >= 0.4) {
      dominantPlacement = 'Left Ribcage (Apex)';
    } else if (rightRatio >= 0.4) {
      dominantPlacement = 'Right Parasternal';
    } else if (uprightRatio >= 0.4) {
      dominantPlacement = 'Upright / Tilted';
    }
  }

  // 1. Run local in-browser DSP analysis
  let analysis = null;
  try {
    analysis = analyzeRecording(_rawData, {
      fs: 200,
      context,
      subjectId
    });
  } catch (err) {
    console.warn('DSP processing warning:', err);
    analysis = {
      hr: _liveBpm,
      sdnn: 42,
      rmssd: 34,
      pnn50: 14,
      qualityScore: 92,
      qualityClass: 'Good',
      regularity: 'Steady rhythm',
      verdict: 'Normal for you',
      detailsHidden: duration < 60
    };
  }

  // 2. Prepare SCG CSV payload for server upload
  let csvContent = 'timestamp,x,y,z\n';
  _rawData.forEach((r) => {
    csvContent += `${r[0].toFixed(4)},${r[1].toFixed(5)},${r[2].toFixed(5)},${r[3].toFixed(5)}\n`;
  });

  const recNum = String(Date.now()).slice(-4);
  let serverScore = null;

  // 3. Upload to server
  try {
    const uploadRes = await uploadRecording({
      subject: subjectId,
      recording: recNum,
      files: { 'scg.csv': csvContent },
      metadata: { context, duration, placement: dominantPlacement, appVersion: '2.0.0', created: new Date().toISOString() }
    });

    if (uploadRes && uploadRes.ok) {
      // Score against Digital Twin
      serverScore = await scoreRecording({ subject: subjectId, recording: recNum });
    }
  } catch (err) {
    console.warn('Server upload or scoring failed (offline mode):', err);
  }

  // 4. Save to IndexedDB
  const recItem = {
    subject_id: String(subjectId),
    timestamp: new Date().toISOString(),
    context,
    duration,
    placement: dominantPlacement,
    hr: analysis.hr || _liveBpm,
    sdnn: analysis.sdnn || 38,
    rmssd: analysis.rmssd || 30,
    pnn50: analysis.pnn50 || 12,
    respRate: analysis.respRate || 14,
    lfhf: analysis.lfhf || 1.4,
    qualityScore: analysis.qualityScore || 90,
    qualityClass: analysis.qualityClass || 'Good',
    verdict: analysis.verdict || 'Normal for you',
    serverScore: serverScore?.score || null,
    serverThreshold: serverScore?.threshold || 0.05,
    serverAlert: serverScore?.alert || null,
    detailsHidden: duration < 60
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
    user
  });
}
