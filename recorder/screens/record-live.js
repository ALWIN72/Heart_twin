/**
 * screens/record-live.js - Step 2: Live Recording & Real-Time Quality Coach
 */

import { drawLiveWaveform } from '../ui/charts.js';

let _motionListener = null;
let _rafId = null;
let _timerInterval = null;
let _rawData = [];
let _liveBuffer = [];
let _startTime = 0;
let _targetDuration = 60;
let _poorSignalDuration = 0;

export function renderRecordLive(container, { context = 'Resting', duration = 60, onComplete, onCancel } = {}) {
  _targetDuration = duration;
  _rawData = [];
  _liveBuffer = new Array(80).fill(0);
  _poorSignalDuration = 0;

  const circumference = 2 * Math.PI * 46; // r=46 -> ~289

  container.innerHTML = `
    <div style="display:flex; flex-direction:column; align-items:center; text-align:center; gap: var(--sp-4);">
      <!-- Top Status Header -->
      <div style="width:100%; display:flex; justify-content:space-between; align-items:center;">
        <span class="badge badge-neutral" id="live-context-badge">${context}</span>
        <span class="badge badge-normal" id="live-quality-badge">● Measuring</span>
      </div>

      <!-- Circular Countdown Ring -->
      <div style="position:relative; width:140px; height:140px; margin: 12px 0;">
        <svg viewBox="0 0 100 100" width="100%" height="100%" style="transform: rotate(-90deg);">
          <circle cx="50" cy="50" r="46" fill="none" stroke="var(--border-subtle)" stroke-width="6" />
          <circle id="countdown-circle" cx="50" cy="50" r="46" fill="none" stroke="var(--primary)" stroke-width="6" 
                  stroke-linecap="round" stroke-dasharray="${circumference}" stroke-dashoffset="0" 
                  style="transition: stroke-dashoffset 1s linear;" />
        </svg>
        <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
          <span style="font-size:32px; font-weight:700; font-family:var(--font-mono); line-height:1;" id="countdown-sec">${duration}</span>
          <span style="font-size:11px; color:var(--text-muted); margin-top:2px;">seconds left</span>
        </div>
      </div>

      <!-- Live Oscilloscope Waveform -->
      <div class="card" style="width:100%; padding: var(--sp-3); overflow:hidden;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-size:11px; font-weight:600; text-transform:uppercase; color:var(--text-muted); letter-spacing:0.04em;">Live Heart Motion</span>
          <span style="font-size:11px; color:var(--text-faint); font-family:var(--font-mono);" id="sensor-rate-display">200 Hz target</span>
        </div>
        <canvas id="live-waveform-canvas" style="width:100%; height:90px; border-radius:var(--r-sm);"></canvas>
      </div>

      <!-- Real-Time Coaching Tip -->
      <div id="inline-coaching-tip" class="hidden card card-sunken" style="text-align:left; border-color:var(--status-slight-border); background:var(--status-slight-bg); padding:10px 14px;">
        <strong style="color:var(--status-slight-text); font-size:13px;">💡 Steady Signal Tip</strong>
        <p style="font-size:12px; color:var(--text-muted); margin-top:2px;">
          Try breathing normally and relaxing your chest muscles without holding your breath.
        </p>
      </div>

      <p style="font-size:13px; color:var(--text-muted);">
        Keep the phone resting flat on your breastbone until the timer completes.
      </p>

      <!-- Cancel / Stop Action -->
      <div style="width:100%; margin-top: auto;">
        <button class="btn btn-secondary btn-block" id="stop-recording-btn" style="color:var(--status-diff-text);">
          Stop Early
        </button>
      </div>
    </div>
  `;

  const canvas = container.querySelector('#live-waveform-canvas');
  const secEl = container.querySelector('#countdown-sec');
  const circleEl = container.querySelector('#countdown-circle');
  const qualityBadge = container.querySelector('#live-quality-badge');
  const rateDisplay = container.querySelector('#sensor-rate-display');
  const tipEl = container.querySelector('#inline-coaching-tip');
  const stopBtn = container.querySelector('#stop-recording-btn');

  _startTime = performance.now();
  let remainingSec = duration;
  let hasSentHalfwayVibe = false;

  // Initial vibration cue
  if ('vibrate' in navigator) navigator.vibrate(80);

  // 1. Motion Listener
  let lastT = 0;
  _motionListener = (event) => {
    const acc = event.accelerationIncludingGravity || event.acceleration;
    if (!acc) return;
    const now = performance.now();
    const t = (now - _startTime) / 1000;
    const x = acc.x || 0;
    const y = acc.y || 0;
    const z = acc.z || 0;

    _rawData.push([t, x, y, z]);

    // Update real-time buffer
    const mag = Math.hypot(x, y, z);
    _liveBuffer.shift();
    _liveBuffer.push(mag - 9.8); // subtract 1g baseline

    // Compute effective frequency
    if (lastT > 0 && _rawData.length % 50 === 0) {
      const hz = Math.round(50 / ((now - lastT) / 1000));
      if (rateDisplay) rateDisplay.textContent = `${hz} Hz sensor`;
    }
    if (_rawData.length % 50 === 0) lastT = now;
  };

  // Check support, fallback to simulated heartbeat if testing on desktop
  let hasPhysicalSensor = false;
  window.addEventListener('devicemotion', (e) => {
    if (e.accelerationIncludingGravity || e.acceleration) {
      hasPhysicalSensor = true;
    }
  }, { once: true });

  window.addEventListener('devicemotion', _motionListener);

  // Desktop test simulation fallback
  let simInterval = null;
  setTimeout(() => {
    if (!hasPhysicalSensor && _rawData.length < 5) {
      if (rateDisplay) rateDisplay.textContent = 'Demo test stream';
      simInterval = setInterval(() => {
        const now = performance.now();
        const t = (now - _startTime) / 1000;
        // Cardiac pulse simulation at ~70 BPM
        const pulse = Math.exp(-Math.pow((t % 0.85) - 0.12, 2) / 0.003) * 0.45;
        const resp = Math.sin(t * 0.25 * Math.PI * 2) * 0.08;
        const noise = (Math.random() - 0.5) * 0.04;
        const z = 9.8 + pulse + resp + noise;
        _rawData.push([t, 0.02, 0.03, z]);
        _liveBuffer.shift();
        _liveBuffer.push(pulse + noise);
      }, 10);
    }
  }, 400);

  // 2. Animation Frame Loop for Canvas Waveform
  function renderLoop() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark' ||
      window.matchMedia('(prefers-color-scheme: dark)').matches;
    drawLiveWaveform(canvas, _liveBuffer, isDark ? 'dark' : 'light');
    _rafId = requestAnimationFrame(renderLoop);
  }
  _rafId = requestAnimationFrame(renderLoop);

  // 3. Countdown Timer Loop
  _timerInterval = setInterval(() => {
    remainingSec--;
    if (secEl) secEl.textContent = Math.max(0, remainingSec);

    // Update Circle Progress
    if (circleEl) {
      const progress = (duration - remainingSec) / duration;
      const offset = circumference * (1 - progress);
      circleEl.style.strokeDashoffset = offset;
    }

    // Halfway vibration cue
    if (!hasSentHalfwayVibe && remainingSec <= duration / 2) {
      hasSentHalfwayVibe = true;
      if ('vibrate' in navigator) navigator.vibrate([50, 50, 50]);
    }

    // Check Quality
    const recent = _liveBuffer.slice(-30);
    const varSignal = recent.reduce((sum, v) => sum + Math.abs(v), 0) / (recent.length || 1);
    if (varSignal > 1.8) {
      _poorSignalDuration++;
      if (qualityBadge) {
        qualityBadge.className = 'badge badge-slight';
        qualityBadge.textContent = '⚠️ Hold still';
      }
    } else {
      _poorSignalDuration = Math.max(0, _poorSignalDuration - 1);
      if (qualityBadge) {
        qualityBadge.className = 'badge badge-normal';
        qualityBadge.textContent = '● Good signal';
      }
    }

    // 10s Coaching Tip
    if (_poorSignalDuration >= 10 && tipEl) {
      tipEl.classList.remove('hidden');
    }

    // Complete!
    if (remainingSec <= 0) {
      cleanup();
      // Victory haptic cue
      if ('vibrate' in navigator) navigator.vibrate([120, 80, 200]);

      if (typeof onComplete === 'function') {
        onComplete({
          context,
          duration,
          rawData: _rawData,
        });
      }
    }
  }, 1000);

  function cleanup() {
    clearInterval(_timerInterval);
    if (simInterval) clearInterval(simInterval);
    cancelAnimationFrame(_rafId);
    window.removeEventListener('devicemotion', _motionListener);
  }

  if (stopBtn) {
    stopBtn.onclick = () => {
      cleanup();
      if (typeof onCancel === 'function') onCancel();
    };
  }
}

